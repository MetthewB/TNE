
from itertools import product
from datetime import datetime
from tqdm import tqdm
import numpy as np

from mjid import MjID
from myoutils import generate_plot, VideoGenerator

# model_path = './myo_sim/arm/myoarm.xml'
my_path = '/Users/nicolegiannotto/Desktop/PaperEBS/code/MyoMjID-main/'
model_path = my_path + 'myo_sim/arm/myoarm.xml'

test_kpkv = False
plot_qfrc = False
gen_video = True

if __name__ == '__main__':
    mjid = MjID(model_path=model_path, disable_contacts=True)

    # shorter names, better for plot titles
    if "leg" in model_path:
        mjid.joint_names[4] = "kr_translation2"
        mjid.joint_names[5] = "kr_translation1"
        mjid.joint_names[7] = "kr_rotation2"
        mjid.joint_names[8] = "kr_rotation3"
        mjid.joint_names[12] = "kr_b_translation2"
        mjid.joint_names[13] = "kr_b_translation1"
        mjid.joint_names[14] = "kr_b_rotation1"
        mjid.joint_names[18] = "kl_translation2"
        mjid.joint_names[19] = "kl_translation1"
        mjid.joint_names[21] = "kl_rotation2"
        mjid.joint_names[22] = "kl_rotation3"
        mjid.joint_names[26] = "kl_b_translation2"
        mjid.joint_names[27] = "kl_b_translation1"
        mjid.joint_names[28] = "kl_b_rotation1"
    elif "arm" in model_path:
        for idx, name in enumerate(mjid.joint_names):
            if "clavicular" in name:
                splitted = name.split("clavicular")
                mjid.joint_names[idx] = splitted[0]+"clav"+splitted[1]

    # indexes of the joints of interest for inverse dynamics
    qidxs_of_interest = [11,12,13,14,15,16,17,18,19,20,21,22,30]

    # target trajectory
    traj = np.genfromtxt(my_path + 'myoarm_sintraj.csv', delimiter=',')
    if np.isnan(traj[0,:][0]):
        traj = traj[1:,:]
    time = traj[:,0]
    traj = traj[:,1:]
    traj[:,17] += (np.sin(time*np.pi-np.pi/2)/4+np.pi/8) # wrist flexion
    traj[:,16] = mjid.model.jnt_range[16].mean() # wrist deviation
    traj[:,18] = mjid.model.jnt_range[18].mean() # cmc abduction
    traj[:,19] = mjid.model.jnt_range[19].mean() # cmc flexion
    traj[:,20] = mjid.model.jnt_range[20].mean() # mp flexion
    traj[:,21] = mjid.model.jnt_range[21].mean() # ip flexion
    traj[:,30] = traj[:,22]

    # starting position
    qpos0 = traj[0,:]

    # main loop to test parameter combinations
    kp = 500; kv = 50; re = 20; rs = 200
    for (rr_s, rr_e, rr_w, rr_t, rr_f) in product([1], [100], [100], [100_000], [10_000]):
        outname = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{kp}_{kv}_{rr_s}_{rr_e}_{rr_w}_{rr_t}_{rr_f}_{re}_{rs}'
        print(f'Working on {outname}...')

        kp2set = kp * np.ones(mjid.model.nq)
        kv2set = kv * np.ones(mjid.model.nq)
        kp2set[[15,16,17,18,19,20,21,22,30]] = 5000
        mjid.update_gains(kp=kp2set, kv=kv2set)

        rr2set = np.ones(mjid.model.nq)
        rr2set[[11,12,13]] = rr_s
        rr2set[[14]] = rr_e
        rr2set[[15,16,17]] = rr_w
        rr2set[[18,19,20,21]] = rr_t
        rr2set[[22,30]] = rr_f
        mjid.update_regs(rr=rr2set, re=re, rs=rs)

        mjid.reset_data(qpos0)

        all_ctrl_achiev = np.zeros((traj.shape[0], mjid.model.nu))
        all_qfrc_target = np.zeros((traj.shape[0], mjid.model.nq))
        all_qfrc_achiev = np.zeros((traj.shape[0], mjid.model.nq))
        all_qpos_achiev = np.zeros((traj.shape[0], mjid.model.nq))

        if gen_video:
            vg = VideoGenerator(mjid.model, mjid.data, outname)

        for idx in tqdm(range(traj.shape[0])):
            qpos_target = traj[idx,:]
            if test_kpkv:
                actuation = mjid.get_qfrc(qpos_target, qidxs_of_interest)
                qfrc_target = actuation
            else:
                actuation, qfrc_target = mjid.get_ctrl(qpos_target, qidxs_of_interest)
                all_ctrl_achiev[idx,:] = actuation
            qpos_achiev, qfrc_achiev = mjid.step_forward(actuation)

            all_qfrc_target[idx,:] = qfrc_target
            all_qfrc_achiev[idx,:] = qfrc_achiev
            all_qpos_achiev[idx,:] = qpos_achiev

            if gen_video:
                vg.step(qpos_target, idx)

        if gen_video:
            vg.release()

        generate_plot(traj, all_qpos_achiev, 
                      all_qfrc_target,
                      all_qfrc_achiev, 
                      all_ctrl_achiev, 
                      mjid.model.jnt_range, 
                      mjid.joint_names, mjid.muscle_names,
                      outname,
                      plot_qfrc=(test_kpkv or plot_qfrc),
                      qidxs_of_interest=qidxs_of_interest)