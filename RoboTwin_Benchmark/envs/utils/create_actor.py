import sapien.core as sapien
import numpy as np
import transforms3d as t3d
import sapien.physx as sapienp
import json

# create box
def create_box(
    scene: sapien.Scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeBox(
            half_size=half_size, material=scene.default_physical_material
        )
    )

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(
            half_size, sapien.render.RenderMaterial(base_color=[*color[:3], 1])
        )
    )

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)

    return entity


# create cylinder
def create_cylinder(
    scene: sapien.Scene,
    pose: sapien.Pose,
    radius: float,
    half_length: float,
    color=None,
    name="",
) -> sapien.Entity:
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeCylinder(
            radius=radius, half_length = half_length, material=scene.default_physical_material
        )
    )

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeCylinder(
            radius=radius, half_length = half_length, material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])
        )
    )

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity

# create box
def create_visual_box(
    scene: sapien.Scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(
            half_size, sapien.render.RenderMaterial(base_color=[*color[:3], 1])
        )
    )

    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity

def create_table(
    scene: sapien.Scene,
    pose: sapien.Pose,
    length: float,
    width: float,
    height: float,
    thickness=0.1,
    color=(1, 1, 1), 
    name="table",
    is_static = True
) -> sapien.Entity:
    """Create a table with specified dimensions."""
    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    # Tabletop
    tabletop_pose = sapien.Pose([0.0, 0.0, -thickness / 2])  # Center the tabletop at z=0
    tabletop_half_size = [length / 2, width / 2, thickness / 2]
    builder.add_box_collision(pose=tabletop_pose, half_size=tabletop_half_size, material=scene.default_physical_material)
    builder.add_box_visual(
        pose=tabletop_pose, half_size=tabletop_half_size, material=color
    )

    # Table legs (x4)
    leg_spacing = 0.1
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (length / 2 - leg_spacing / 2) 
            y = j * (width / 2 - leg_spacing / 2)
            table_leg_pose = sapien.Pose([x, y, -height / 2])
            table_leg_half_size = [thickness / 2, thickness / 2, height / 2]
            builder.add_box_collision(
                pose=table_leg_pose, half_size=table_leg_half_size
            )
            builder.add_box_visual(
                pose=table_leg_pose, half_size=table_leg_half_size, material=color
            )

    table = builder.build(name=name)
    table.set_pose(pose)
    return table

# create obj model
def create_obj(
    scene: sapien.Scene,
    pose: sapien.Pose,
    modelname: str,
    scale = (1,1,1),
    convex = False,
    is_static = False,
    model_id = None,
    model_z_val = False
) -> sapien.Entity:
    modeldir = "./models/"+modelname+"/"
    if model_id is None:
        file_name = modeldir + "textured.obj"
        json_file_path = modeldir + 'model_data.json'
    else:
        file_name = modeldir + f"textured{model_id}.obj"
        json_file_path = modeldir + f'model_data{model_id}.json'

    try:
        with open(json_file_path, 'r') as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None
        
    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if model_z_val:
        pose.set_p(pose.get_p()[:2].tolist() + [0.74 + (t3d.quaternions.quat2mat(pose.get_q()) @ (np.array(model_data["extents"]) * scale))[2]/2])
        
    if convex==True:
        builder.add_multiple_convex_collisions_from_file(
            filename = file_name,
            scale= scale
        )
    else:
        builder.add_nonconvex_collision_from_file(
            filename = file_name,
            scale = scale
        )
    
    builder.add_visual_from_file(
        filename=file_name,
        scale= scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)
    
    return mesh, model_data


# create glb model
def create_glb(
    scene: sapien.Scene,
    pose: sapien.Pose,
    modelname: str,
    scale = (1,1,1),
    convex = False,
    is_static = False,
    model_id = None,
    model_z_val = False
) -> sapien.Entity:
    modeldir = "./models/"+modelname+"/"
    if model_id is None:
        file_name = modeldir + "base.glb"
        json_file_path = modeldir + 'model_data.json'
    else:
        file_name = modeldir + f"base{model_id}.glb"
        json_file_path = modeldir + f'model_data{model_id}.json'
    
    try:
        with open(json_file_path, 'r') as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None
    
    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if model_z_val:
        pose.set_p(pose.get_p()[:2].tolist() + [0.74 + (t3d.quaternions.quat2mat(pose.get_q()) @ (np.array(model_data["extents"]) * scale))[2]/2])

    if convex==True:
        builder.add_multiple_convex_collisions_from_file(
            filename = file_name,
            scale= scale
        )
    else:
        builder.add_nonconvex_collision_from_file(
            filename = file_name,
            scale = scale,
        )
    
    builder.add_visual_from_file(
        filename=file_name,
        scale= scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    return mesh, model_data


def create_actor(
    scene: sapien.Scene,
    pose: sapien.Pose,
    modelname: str,
    scale = (1,1,1),
    convex = False,
    is_static = False,
    model_id = None,
    model_z_val = False
) -> sapien.Entity:
    try:
        return create_glb(
            scene = scene,
            pose = pose,
            modelname = modelname,
            scale = scale,
            convex = convex,
            is_static = is_static,
            model_id = model_id,
            model_z_val = model_z_val
        )
    except:
        try:
            return create_obj(
                scene = scene,
                pose = pose,
                modelname = modelname,
                scale = scale,
                convex = convex,
                is_static = is_static,
                model_id = model_id,
                model_z_val = model_z_val
            )
        except:
            print(modelname, 'is not exsist model file!')
            return None, None



# create urdf model
def create_urdf_obj(
    scene: sapien.Scene,
    pose: sapien.Pose,
    modelname: str,
    scale = 1.0,
    fix_root_link = True
)->sapienp.PhysxArticulation: 
    modeldir = "./models/"+modelname+"/"
    file_name = modeldir + "base.glb"
    json_file_path = modeldir + 'model_data.json'
    
    try:
        with open(json_file_path, 'r') as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    modeldir = "./models/"+modelname+"/"
    object: sapien.Articulation = loader.load(modeldir+"mobility.urdf")
    
    object.set_root_pose(pose)
    return object, model_data