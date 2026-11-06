"""Exercise the real BatchEnvPool with a sensor-callback completion barrier."""
import concurrent.futures
import ctypes
import sys
import threading

import mujoco
import numpy as np

from mujoco.batch_env import BatchEnvPool

print("NATIVE_MODULE", mujoco._batch_env.__file__, flush=True)

model = mujoco.MjModel.from_xml_string("""
<mujoco><option timestep="0.002" gravity="0 0 0"/>
<worldbody><body><joint name="hinge"/><geom type="sphere" size="0.1"/></body></worldbody>
<actuator><motor joint="hinge"/></actuator>
<sensor><jointpos joint="hinge"/><user name="barrier" dim="1" needstage="pos"/></sensor></mujoco>
""")
data = mujoco.MjData(model)
state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
initial = np.tile(state, (8, 1))
entered = threading.Event()
release = threading.Event()


@ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
def sensor_callback(_model, callback_data, stage):
    if stage == mujoco.mjtStage.mjSTAGE_POS:
        entered.set()
        release.wait(10)


with BatchEnvPool(model, nbatch=8, nthread=2) as pool:
    mujoco.set_mjcb_sensor(sensor_callback)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(pool.forward, initial)
            try:
                if not entered.wait(10):
                    raise AssertionError("native worker did not enter sensor callback")
                try:
                    pool.get_field(0, "gravity")
                except RuntimeError as error:
                    assert "overlapping operations" in str(error), str(error)
                else:
                    raise AssertionError("overlapping native operation was admitted")
            finally:
                release.set()
            observed = future.result(timeout=10)
            np.testing.assert_array_equal(observed, np.zeros((8, 2)))
    finally:
        mujoco.set_mjcb_sensor(None)

    fault_library = ctypes.CDLL(sys.argv[1])
    failing_sensor = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)(
        ("hpd_fail_sensor", fault_library)
    )
    mujoco.set_mjcb_sensor(failing_sensor)
    try:
        try:
            pool.forward(initial)
        except mujoco.FatalError as error:
            assert "native worker failure sentinel" in str(error), str(error)
        else:
            raise AssertionError("native worker fatal error was not propagated")
    finally:
        mujoco.set_mjcb_sensor(None)
    np.testing.assert_array_equal(pool.forward(initial), np.zeros((8, 2)))
    print("PASS: native worker fatal error propagated and operation ownership released")

    old_state, old_sensor = pool.step(initial, nstep=2, return_sensor=True)
    saved_state = old_state.copy()
    saved_sensor = old_sensor.copy()
    next_initial = initial.copy()
    next_initial[:, 1] = 0.25
    pool.reset([0, 3], next_initial[[0, 3]])
    forwarded = pool.forward(next_initial)
    np.testing.assert_allclose(forwarded[:, 0], 0.25, rtol=0, atol=1e-12)
    controls = np.full((8, 3, 1), 0.2)
    actual_state, _ = pool.step(next_initial, nstep=3, control=controls, return_sensor=True)
    reference = np.empty_like(actual_state)
    for row in range(8):
        reference_data = mujoco.MjData(model)
        mujoco.mj_setState(model, reference_data, next_initial[row], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        for _ in range(3):
            reference_data.ctrl[0] = 0.2
            mujoco.mj_step(model, reference_data)
        mujoco.mj_getState(model, reference_data, reference[row], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    np.testing.assert_allclose(actual_state, reference, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(old_state, saved_state)
    np.testing.assert_array_equal(old_sensor, saved_sensor)
np.testing.assert_array_equal(old_state, saved_state)
np.testing.assert_array_equal(old_sensor, saved_sensor)
print("PASS: overlap rejected; worker completion, pool reuse, and returned-storage lifetime verified")
print("PASS: native stepping matches the independent serial MuJoCo reference")
