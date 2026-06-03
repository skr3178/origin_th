"""
Render robomimic low_dim demos to video by replaying saved MuJoCo sim states.

The low_dim_v141.hdf5 file stores no images, only low-dim state vectors plus the
full simulator `states` and per-demo `model_file` (MuJoCo XML). We recreate the
robosuite env, load each saved state, and render an offscreen camera to MP4.

Usage:
    MUJOCO_GL=egl python render_demos.py --demos demo_0 demo_1 --camera agentview
"""
import argparse, json, os
import h5py
import numpy as np
import imageio
import robosuite
from robosuite.controllers import load_controller_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/lift/ph/low_dim_v141.hdf5")
    ap.add_argument("--outdir", default="out_viz")
    ap.add_argument("--demos", nargs="*", default=None,
                    help="demo keys to render; default = first N")
    ap.add_argument("--n", type=int, default=3, help="number of demos if --demos unset")
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    f = h5py.File(args.dataset, "r")
    env_args = json.loads(f["data"].attrs["env_args"])
    kwargs = dict(env_args["env_kwargs"])

    # turn on the offscreen renderer + the camera we want
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
    )
    # controller_configs in the file is already a full dict; pass through
    env = robosuite.make(env_args["env_name"], **kwargs)

    demo_keys = args.demos or sorted(f["data"].keys(),
                                     key=lambda k: int(k.split("_")[1]))[: args.n]

    for dk in demo_keys:
        grp = f["data"][dk]
        states = grp["states"][:]
        model_xml = grp.attrs["model_file"]

        env.reset()
        # the stored XML has absolute mesh paths from the original author's
        # machine; edit_model_xml rewrites them to the local robosuite install
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()

        cam_key = f"{args.camera}_image"
        out_path = os.path.join(args.outdir, f"lift_{dk}_{args.camera}.mp4")
        writer = imageio.get_writer(out_path, fps=args.fps)
        for st in states:
            env.sim.set_state_from_flattened(st)
            env.sim.forward()
            obs = env._get_observations(force_update=True)
            img = obs[cam_key][::-1]  # robosuite returns flipped vertically
            writer.append_data(img)
        writer.close()
        print(f"wrote {out_path}  ({len(states)} frames)")

    f.close()


if __name__ == "__main__":
    main()
