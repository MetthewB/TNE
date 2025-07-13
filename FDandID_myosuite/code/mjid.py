
import scipy.sparse as spa
import numpy as np
import mujoco
import osqp

def mju_sigmoid(x):
    return np.clip(x*x*x*(3*x*(2*x-5)+10),0,1)

def solve_qp(P, q, lb, ub, x0):
    P = spa.csc_matrix(P)
    A = spa.csc_matrix(spa.eye(q.shape[0]))
    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=lb, u=ub, verbose=False)
    m.warm_start(x=x0)
    res = m.solve()
    return res.x

class MjID(object):
    def __init__(self, model_path, disable_contacts):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        if disable_contacts:
            self.model.opt.disableflags += mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.data = mujoco.MjData(self.model)
        self.joint_names = [self.model.joint(i).name for i in range(self.model.nv)]
        self.muscle_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        # ----
        self.model.actuator_actearly = 0
        self.model.actuator_dynprm[:,2] = 1 # tausmooth
        self.kp = 1000; self.kv = 100
        self.reg_replic = 1
        self.reg_energy = 0
        self.reg_smooth = 0
        self.reg_der = 0
        # ----
        self._x = np.linspace(0,0.5,5) # x = 0.5 + (ctrl - act) / tausmooth
        self._X = np.vstack([np.ones_like(self._x), self._x]).T
        self._XtX = self._X.T @ self._X

    def update_gains(self, kp, kv):
        self.kp = kp
        self.kv = kv

    def update_regs(self, rr, re, rs, rd):
        self.reg_replic = rr
        self.reg_energy = re
        self.reg_smooth = rs
        self.reg_der = rd

    def reset_data(self, qpos0=0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos = qpos0
        mujoco.mj_forward(self.model, self.data) 
        # mujoco.mj_step1(self.model, self.data)

    def step_forward(self, actuation):
        if len(actuation) == self.model.nv:
            self.data.qfrc_applied = actuation
        else:
            self.data.ctrl = actuation
        mujoco.mj_step(self.model, self.data)
        return self.data.qpos.copy(), self.data.qfrc_actuator.copy()
    
    def get_qfrc(self, qpos_target, qidxs_of_interest=None, qfrc_contacts=0):
        if qidxs_of_interest is None:
            qidxs_of_interest = list(range(len(qpos_target)))
        yolo = self.data.qpos.copy(); 
        yolo[qidxs_of_interest] = qpos_target[qidxs_of_interest]
        qpos_target = np.clip(yolo, self.model.jnt_range[:,0], self.model.jnt_range[:,1])
        self.data.qacc = self.kp * (qpos_target - self.data.qpos) - self.kv * self.data.qvel
        self.model.opt.disableflags += mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        mujoco.mj_inverse(self.model, self.data)
        self.model.opt.disableflags -= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        qfrc_target = self.data.qfrc_inverse - qfrc_contacts
        
        return qfrc_target

    def get_ctrl(self, qpos_target, qidxs_of_interest=None, qfrc_contacts=0, ctrl_ref=0):
        # ---- get qfrc
        qfrc_target = self.get_qfrc(qpos_target, qidxs_of_interest, qfrc_contacts)
        # ---- get taus
        tau_act =   self.model.actuator_dynprm[:,0] * (0.5 + 1.5 * self.data.act)
        tau_deact = self.model.actuator_dynprm[:,1] / (0.5 + 1.5 * self.data.act)
        tausmooth = self.model.actuator_dynprm[:,2]
        # ---- approximate muscle dynamics
        f = lambda x: tausmooth[:,None] * (x - 0.5) / (tau_deact[:,None] + (tau_act[:,None] - tau_deact[:,None]) * mju_sigmoid(x))
        y = f(self._x)
        XtY = self._X.T @ y.T
        coefficients = np.linalg.inv(self._XtX) @ XtY
        b = coefficients[0,:]
        a = coefficients[1,:]
        # ---- get gain, bias
        gain = np.zeros(self.model.nu)
        bias = np.zeros(self.model.nu)
        for idx_actuator in range(self.model.nu):
            length = self.data.actuator_length[idx_actuator]
            lengthrange = self.model.actuator_lengthrange[idx_actuator]
            velocity = self.data.actuator_velocity[idx_actuator]
            acc0 = self.model.actuator_acc0[idx_actuator]
            prmb = self.model.actuator_biasprm[idx_actuator,:9]
            prmg = self.model.actuator_gainprm[idx_actuator,:9]
            bias[idx_actuator] = mujoco.mju_muscleBias(length, lengthrange, acc0, prmb)
            gain[idx_actuator] = mujoco.mju_muscleGain(length, velocity, lengthrange, acc0, prmg)
        # ---- get others
        AM = self.data.actuator_moment.T
        ts = self.model.opt.timestep
        act = self.data.act
        ctrl0 = self.data.ctrl
        d_ctrl0 = abs(np.diff(ctrl0) / ts)
        d_ctrl0 = np.append(d_ctrl0, d_ctrl0[-1]) 
        # ---- get ctrl
        taus = tausmooth[0]
        max_change = 0.5 * taus
        lb = np.clip(act - max_change, 0, 1)
        ub = np.clip(act + max_change, 0, 1)
        k = AM @ (gain * (act * (1 - a * ts / taus) + 0.5 * a * ts + b * ts) + bias) - qfrc_target
        A = AM @ np.diag(gain * a * ts / taus)
        P = 2 * (self.reg_replic * A.T @ A + self.reg_energy * np.eye(A.shape[1]) + self.reg_smooth * np.eye(A.shape[1]) + self.reg_der * np.eye(A.shape[1]))
        q = 2 * (self.reg_replic * k @ A - self.reg_energy * ctrl_ref - self.reg_smooth * ctrl0 - self.reg_der * d_ctrl0)
        ctrl = solve_qp(P, q, lb, ub, ctrl0)
        return np.clip(ctrl, 0, 1), qfrc_target