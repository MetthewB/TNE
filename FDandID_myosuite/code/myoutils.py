from scipy.signal import savgol_filter
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import mujoco
import cv2

AZIMUTH = 90
DISTANCE = 2
ELEVATION = 0
LOOKAT = [0, 0, 1.173]

def generate_plot(all_qpos_target,
                  all_qpos_achiev,
                  all_qfrc_target,
                  all_qfrc_achiev,
                  all_ctrl_achiev,
                  joint_ranges,
                  joint_names,
                  muscle_names,
                  outname,
                  plot_ticks=False,
                  plot_qfrc=False,
                  qidxs_of_interest=None,
                  all_qpos_achiev_test=None,
                  all_emg_data=None
                  ):
    timestamps = list(range(all_qpos_target.shape[0]))
    nq = all_qpos_achiev.shape[1]
    nu = all_ctrl_achiev.shape[1]
    cols = 11
    mul = (nq//cols) + 1 if nq % cols else (nq//cols)
    n_plots_for_q = mul * cols
    mul = (nu//cols) + 1 if nu % cols else (nu//cols)
    n_plots_for_u = mul * cols if all_ctrl_achiev.any() else 0
    total_subplots = 2 * (n_plots_for_q) + n_plots_for_u if plot_qfrc else n_plots_for_q + n_plots_for_u
    rows = int(np.ceil(total_subplots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.5*cols, 1.5*rows))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < nq:
            linestyle = '--' if qidxs_of_interest is not None and i not in qidxs_of_interest else '-'
            ax.plot(timestamps, all_qpos_achiev[:,i], linestyle=linestyle)
            if all_qpos_achiev_test is not None:
                ax.plot(timestamps, all_qpos_achiev_test[:,i], linestyle=linestyle, color='r') 
            if i in qidxs_of_interest:
                ax.plot(timestamps, all_qpos_target[:,i], linestyle=linestyle)
            ax.set_xlim(timestamps[0],timestamps[-1])
            ax.axhline(joint_ranges[i][0], color='r', linestyle='--')
            ax.axhline(joint_ranges[i][1], color='r', linestyle='--')
            ax.set_title(joint_names[i], fontsize='small')
        elif plot_qfrc and n_plots_for_q <= i < n_plots_for_q + nq:
            linestyle = '--' if qidxs_of_interest is not None and i-n_plots_for_q not in qidxs_of_interest else '-'
            ax.plot(timestamps, all_qfrc_achiev[:,i-n_plots_for_q], linestyle=linestyle)
            ax.plot(timestamps, all_qfrc_target[:,i-n_plots_for_q], linestyle=linestyle)
            ax.set_xlim(timestamps[0],timestamps[-1])
            ax.set_title(joint_names[i-n_plots_for_q], fontsize='small')
        elif (plot_qfrc and 2 * n_plots_for_q <= i < 2 * n_plots_for_q + nu) or (not plot_qfrc and n_plots_for_q <= i < n_plots_for_q + nu):
            offset = 2 * n_plots_for_q if plot_qfrc else n_plots_for_q
            ax.plot(timestamps, all_ctrl_achiev[:,i-offset], color='tab:pink')
            if all_emg_data is not None:
                ax.plot(timestamps, all_emg_data[:,i-offset], color='tab:cyan', linestyle='--')
            ax.set_xlim(timestamps[0],timestamps[-1])
            ax.set_ylim(0,1)
            ax.set_title(muscle_names[i-offset], fontsize='small')
        else:
            ax.axis('off')
        if not plot_ticks:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(bottom=False, top=False, left=False, right=False)
    legend_index = nq + 2 # + 3
    legend_elements = [
        Line2D([0], [0], linestyle='--', color='r',          label='Joint ranges'),
        Line2D([0], [0], linestyle='--', color='tab:blue',   label='Achieved trajectory without reference'),
        Line2D([0], [0], linestyle='-',  color='tab:blue',   label='Achieved trajectory with reference'),
        Line2D([0], [0], linestyle='-',  color='tab:orange', label='Reference trajectory'),
        Line2D([0], [0], linestyle='-',  color='tab:pink',   label='Achieved muscle activations')
    ]
    if all_emg_data is not None:
        legend_elements.append(Line2D([0], [0], linestyle='--',  color='tab:cyan',   label='sEMG envelopes'))
    axes[legend_index].legend(handles=legend_elements, loc='center', bbox_to_anchor=(0, 0.5), ncol=2, bbox_transform=axes[legend_index].transAxes, fontsize='small')
    plt.savefig(outname+'.png', bbox_inches='tight')
    plt.close()

def do_interp(sig, t, t_new):
    def interp(y_old, x_old, x_new):
        y_new = np.interp(x_new, x_old, y_old)
        return y_new
    sig_i = np.apply_along_axis(interp, 0, sig, t, t_new)
    return sig_i

def do_savgol_filt(sig, winlen, polyorder=1):
    sig_f = np.apply_along_axis(savgol_filter, 0, sig, winlen, polyorder)
    return sig_f

class VideoGenerator(object):
    def __init__(self, mj_model, mj_data, outname):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.data_ref = mujoco.MjData(mj_model)
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = AZIMUTH
        self.camera.distance = DISTANCE
        self.camera.elevation = ELEVATION
        self.camera.lookat = LOOKAT
        # ----
        self.options_ref = mujoco.MjvOption()
        self.options_ref.flags[4] = 0 # actuator OFF
        self.options_ref.flags[7] = 0 # tendon OFF
        self.options_ref.flags[22] = 0 # static body OFF
        self.options_ref.geomgroup[1:] = 0 # body OFF
        self.options_ref.sitegroup[:] = 0
        # ----
        self.options_test = mujoco.MjvOption()
        self.options_test.flags[4] = 1 # actuator ON
        # self.options_test.flags[22] = 0 # static body OFF
        self.options_test.geomgroup[1:] = 0 # body OFF
        self.options_test.sitegroup[:] = 0
        # ----
        self.renderer_ref = mujoco.Renderer(mj_model, height=480, width=480)
        self.renderer_ref.scene.flags[:] = 0
        self.renderer_test = mujoco.Renderer(mj_model, height=480, width=480)
        self.renderer_test.scene.flags[:] = 0
        # ----
        self.reset(outname)

    def reset(self, outname):
        self.fps = 25; tuple_size = (960,480)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(outname+'.mp4', fourcc, self.fps, tuple_size)

    def step(self, qpos_target, idx):
        self.data_ref.qpos = qpos_target
        mujoco.mj_step1(self.mj_model, self.data_ref)

        if not idx % round(1/(self.mj_model.opt.timestep*self.fps)):
            self.renderer_ref.update_scene(self.data_ref, camera=self.camera, scene_option=self.options_ref)
            frame_ref = self.renderer_ref.render()
            self.renderer_test.update_scene(self.mj_data, camera=self.camera, scene_option=self.options_test)
            frame_test = self.renderer_test.render()
            frame_merged = np.append(frame_ref, frame_test, axis=1)
            self.out.write(cv2.cvtColor(frame_merged, cv2.COLOR_RGB2BGR))

    def release(self):
        self.out.release()
