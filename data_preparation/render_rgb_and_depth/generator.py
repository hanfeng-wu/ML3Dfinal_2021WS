import bpy
import bpy_extras
from mathutils import Matrix
from mathutils import Vector
from mathutils import Euler
from mathutils import Quaternion
from pathlib import Path
import os
import numpy as np
import bmesh
import math
import shutil

output_dir = Path(os.getcwd()) / "output/"
input_dir = Path(os.getcwd()) / "input/"
shapenet_dir = Path("D:/Code/shapenet/ShapeNetCore.v2/04379243")
cam_distance = 1.5
cam_direction = (0.669, -0.61, 0.42)

cam_settings = (
    (Vector((   1.0035,    -0.915,      0.63)), Quaternion((0.780483, 0.483536,  0.208704,  0.336872))), # standard
    (Vector((        0,      -1.5,         0)), Quaternion((0.707107, 0.707107,       0.0,       0.0))), # back
    (Vector((        0,       1.5,         0)), Quaternion((     0.0,      0.0,  0.707107,  0.707107))), # front
    (Vector((      1.5,         0,         0)), Quaternion((    -0.5,     -0.5,      -0.5,      -0.5))), # side
    (Vector(( 0.844254,  0.935474, -0.813709)), Quaternion((0.171661,  0.31519,   0.81969,  0.446426))), # bottom left
    (Vector((  1.33636, -0.281097, -0.620576)), Quaternion((0.420406, 0.652824,  0.529792,  0.341175))), # bottom right
    (Vector((-0.006254,   0.89002,    1.2074)), Quaternion((0.003338, 0.001097, -0.312299, -0.949977))), # front top
    (Vector((-0.009829,  -1.41057,   -0.5101)), Quaternion((0.574424, 0.818551, -0.002852, -0.002001))), # behind bottom
    (Vector(( 0.909868,  0.960473,  0.706847)), Quaternion((0.317473, 0.190327,  0.477661,   0.79676))), # top right
    (Vector((-0.833677, -0.506536,    2.3928)), Quaternion((0.855419, 0.168492, -0.09465, -0.480532))), # top left
)

depth_file_slot = None
rgb_file_slot = None

def basic_setup():
    bpy.context.scene.cycles.device = 'GPU'
    # bpy.context.scene.render.image_settings.file_format='PNG'

    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    # bpy.context.scene.render.engine = 'CYCLES'
    # bpy.context.scene.cycles.use_denoising = True
    # bpy.context.scene.cycles.use_adaptive_sampling = True

    bpy.context.scene.render.film_transparent = True
    bpy.data.objects["Camera"].location = [cam_distance * c for c in cam_direction]
    # bpy.data.objects["Camera"].location       = (0.992778, -0.478136, 1.61844)
    # bpy.data.objects["Camera"].rotation_euler = Euler((math.radians(-32.3006), math.radians(13.4723), math.radians(-90)))


def setup_scene():
    objs = bpy.data.objects
    for obj in objs:
        if obj.type == "MESH":
            objs.remove(obj, do_unlink=True)


def set_env_map(path):
    # Get the environment node tree of the current scene
    tree = bpy.context.scene.world.node_tree

    # Clear all nodes
    tree.nodes.clear()

    # Add Background node
    node_background = tree.nodes.new(type='ShaderNodeBackground')

    # Add Environment Texture node
    node_environment = tree.nodes.new('ShaderNodeTexEnvironment')
    # Load and assign the image to the node property
    node_environment.image = bpy.data.images.load(path) # Relative path
    node_environment.location = -300,0

    # Add Output node
    node_output = tree.nodes.new(type='ShaderNodeOutputWorld')
    node_output.location = 200,0

    # Link all nodes
    links = tree.links
    link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

def enable_depth():
    global rgb_file_slot
    global depth_file_slot

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    tree.nodes.clear()
    render_layers   = tree.nodes.new(type='CompositorNodeRLayers')
    output_file_d   = tree.nodes.new(type='CompositorNodeOutputFile')
    output_file_rgb = tree.nodes.new(type='CompositorNodeOutputFile')
    normalize       = tree.nodes.new(type='CompositorNodeNormalize')

    #output_file_d.base_path = str(output_dir)
    output_file_d.format.file_format = "OPEN_EXR"
    output_file_d.file_slots.values()[0].path = "depth"
    depth_file_slot = output_file_d

    # output_file_rgb.base_path = str(output_dir)
    output_file_rgb.format.file_format = "OPEN_EXR"
    rgb_file_slot = output_file_rgb
    output_file_rgb.file_slots.values()[0].path = "rgb"

    links = tree.links
    link = links.new(render_layers.outputs["Image"], output_file_rgb.inputs["Image"])
    # link = links.new(render_layers.outputs["Depth"], normalize.inputs["Value"])
    # link = links.new(normalize.outputs["Value"], output_file_d.inputs["Image"])
    link = links.new(render_layers.outputs["Depth"], output_file_d.inputs["Image"])

# NOTE(Felix): source: https://github.com/DLR-RM/BlenderProc/blob/5be0cb1257d83e22535afba739bd6f9b359f4c03/blenderproc/python/camera/CameraUtility.py#L209
def get_view_fac_in_px(cam: bpy.types.Camera, pixel_aspect_x: float, pixel_aspect_y: float,
                       resolution_x_in_px: int, resolution_y_in_px: int) -> int:
    """ Returns the camera view in pixels.
    :param cam: The camera object.
    :param pixel_aspect_x: The pixel aspect ratio along x.
    :param pixel_aspect_y: The pixel aspect ratio along y.
    :param resolution_x_in_px: The image width in pixels.
    :param resolution_y_in_px: The image height in pixels.
    :return: The camera view in pixels.
    """
    # Determine the sensor fit mode to use
    if cam.sensor_fit == 'AUTO':
        if pixel_aspect_x * resolution_x_in_px >= pixel_aspect_y * resolution_y_in_px:
            sensor_fit = 'HORIZONTAL'
        else:
            sensor_fit = 'VERTICAL'
    else:
        sensor_fit = cam.sensor_fit

    # Based on the sensor fit mode, determine the view in pixels
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px

    return view_fac_in_px

# NOTE(Felix): source: https://github.com/DLR-RM/BlenderProc/blob/5be0cb1257d83e22535afba739bd6f9b359f4c03/blenderproc/python/camera/CameraUtility.py#L196
def get_sensor_size(cam: bpy.types.Camera) -> float:
    """ Returns the sensor size in millimeters based on the configured sensor_fit.
    :param cam: The camera object.
    :return: The sensor size in millimeters.
    """
    if cam.sensor_fit == 'VERTICAL':
        sensor_size_in_mm = cam.sensor_height
    else:
        sensor_size_in_mm = cam.sensor_width
    return sensor_size_in_mm


# NOTE(Felix): source: https://github.com/DLR-RM/BlenderProc/blob/main/blenderproc/python/camera/CameraUtility.py#L239
def get_intrinsic_mat():
    """ Returns the current set intrinsics in the form of a K matrix.
    This is basically the inverse of the the set_intrinsics_from_K_matrix() function.
    :return: The 3x3 K matrix
    """
    cam_ob = bpy.context.scene.camera
    cam = cam_ob.data

    f_in_mm = cam.lens
    resolution_x_in_px = bpy.context.scene.render.resolution_x
    resolution_y_in_px = bpy.context.scene.render.resolution_y

    # Compute sensor size in mm and view in px
    pixel_aspect_ratio = bpy.context.scene.render.pixel_aspect_y / bpy.context.scene.render.pixel_aspect_x
    view_fac_in_px = get_view_fac_in_px(cam, bpy.context.scene.render.pixel_aspect_x, bpy.context.scene.render.pixel_aspect_y, resolution_x_in_px, resolution_y_in_px)
    sensor_size_in_mm = get_sensor_size(cam)

    # Convert focal length in mm to focal length in px
    fx = f_in_mm / sensor_size_in_mm * view_fac_in_px
    fy = fx / pixel_aspect_ratio

    # Convert principal point in blenders format to px
    cx = (resolution_x_in_px - 1) / 2 - cam.shift_x * view_fac_in_px
    cy = (resolution_y_in_px - 1) / 2 + cam.shift_y * view_fac_in_px / pixel_aspect_ratio

    # Build K matrix
    K = Matrix(((fx,  0, cx),
                ( 0, fy, cy),
                ( 0,  0,  1)))
    return list(list(row) for row in K)

# NOTE(Felix): source: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_extrinsic_mat(location, rotation):
    # cam = bpy.data.objects['Camera']

    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1,  0),
        (0,  0, -1)))

    # location, rotation = cam.matrix_world.decompose()[0:2]
    # print("ROT::: ", rotation)
    R_world2bcam = rotation.to_matrix().transposed()

    T_world2bcam = -1*R_world2bcam @ location

    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],),
        (0,0,0,1),
        ))

    return list([round(e, 4) for e in row] for row in RT)


basic_setup()
set_env_map(str(Path(os.getcwd()) / "hdris/shanghai_bund_4k.exr"))
enable_depth()

target_objs_from = 0
target_objs_to = 2000
for idx, obj_dir in enumerate(shapenet_dir.iterdir()):
    if idx < target_objs_from:
        continue
    if idx == target_objs_to:
        break

    setup_scene()
    obj_path = obj_dir / "models" / "model_normalized.obj"

    obj_name = obj_dir.parts[-1]

    rgb_file_slot.base_path                = str(output_dir / obj_name)
    depth_file_slot.base_path              = str(output_dir / obj_name)

    bpy.ops.import_scene.obj(filepath=str(obj_path))

    objs = bpy.data.objects
    for obj in objs:
        if obj.type == "CAMERA":
            camera = obj

    for c_idx, cam_setting in enumerate(cam_settings):
        bpy.context.scene.frame_set(c_idx)

        camera.rotation_mode = 'QUATERNION'
        camera.rotation_quaternion = cam_setting[1]
        camera.location = cam_setting[0]

        bpy.ops.render.render()

    os.system(f"title {idx} / {target_objs_to} {round(100* (idx-target_objs_from) / (target_objs_to-target_objs_from-1), 2)} %")

    # NOTE(Felix): rename the files blender created to get rid of the current
    #   frame number that blender inserted that we can't turn off
    # (output_dir / (rgb_file_slot.path+"0001.exr")).replace(Path(output_dir)/(rgb_file_slot.path+".exr"))
    # (output_dir / (depth_file_slot.path+"0001.exr")).replace(Path(output_dir)/(depth_file_slot.path+".exr"))

    shutil.copy(obj_path, output_dir / obj_name / f"model.obj")
    shutil.copy(obj_dir / "models" / "model_normalized.solid.binvox"   , output_dir / obj_name / f"model.solid.binvox")
    shutil.copy(obj_dir / "models" / "model_normalized.surface.binvox" , output_dir / obj_name / f"model.surface.binvox")

# bpy.ops.wm.save_as_mainfile(filepath="test.blend")

with open(str(output_dir / "matrices.txt"), "w") as matrix_file:
    print(get_intrinsic_mat(), file=matrix_file)
    print(get_intrinsic_mat())

    for c_idx, cam_setting in enumerate(cam_settings):
        #print(cam_setting)
        camera.rotation_mode = 'QUATERNION'
        camera.rotation_quaternion = cam_setting[1]
        camera.location = cam_setting[0]
        print(get_extrinsic_mat(cam_setting[0], cam_setting[1]))
        print(get_extrinsic_mat(cam_setting[0], cam_setting[1]), file=matrix_file)
