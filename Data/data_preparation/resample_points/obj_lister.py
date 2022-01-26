from pathlib import Path
import os
import errno

in_path = Path(r"D:\Studium\ML43D\rerender_with_depth\output")
our_path = Path(r"D:\Studium\ML43D\rerender_with_depth\rpc")

with open("mesh_dirs.txt", "w") as out_file:
    for obj_dir in in_path.iterdir():
        obj_out_file = our_path / obj_dir.parts[-1] / "pcd_gt.obj"

        if not os.path.exists(os.path.dirname(obj_out_file)):
            try:
                os.makedirs(os.path.dirname(obj_out_file))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        print(str(obj_dir / "model.obj"),
              str(obj_out_file),
              file=out_file)
