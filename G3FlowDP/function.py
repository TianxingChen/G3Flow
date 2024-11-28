import torch
import pickle
import pytorch3d.ops as torch3d_ops
import einops as E
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
import os
import shutil

def save_image(color, output_path, mask=None, mask_color=0):
    parent_dir = os.path.dirname(output_path)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        
    if mask is None:
        image = Image.fromarray(color)
        image.save(output_path, format='PNG')
        return

    assert color.shape[:2] == mask.shape, "unmatch"

    masked_color = color.copy()
    masked_color[~mask] = mask_color

    masked_image = Image.fromarray(masked_color)

    masked_image.save(output_path, format='PNG')


def center_padding(images, patch_size):
    _, _, h, w = images.shape
    diff_h = h % patch_size
    diff_w = w % patch_size

    if diff_h == 0 and diff_w == 0:
        return images

    pad_h = patch_size - diff_h
    pad_w = patch_size - diff_w

    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_b = pad_h - pad_t

    images = F.pad(images, (pad_l, pad_r, pad_t, pad_b))
    return images

def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output

def transform_np_image_to_torch(image, transform_size):
    img = np.array(image)[:, :, :3]
    H, W = img.shape[0], img.shape[1]
    img = Image.fromarray(img)
    rgb_transform = transforms.Compose(
                [
                    transforms.Resize((transform_size, transform_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
    img = rgb_transform(img).to('cuda')
    img = img.unsqueeze(0).detach()
    return img, H, W

def transform_shape(x, H, W):
    tmp = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
    return tmp[0].permute(1, 2, 0)

    
def PCA_visualize(feature, H, W, return_res=False, pcaed=False, mask=None):
    feature_img_resized = F.interpolate(feature, 
                            size=(H, W), 
                            mode='bilinear', 
                            align_corners=True)
    feature_img_resized = feature_img_resized[0].permute(1, 2, 0)
    feature = feature_img_resized
    if feature.device != torch.device('cpu'):
        feature = feature.cpu()

    if not pcaed:
        pca = PCA(n_components=3)
        tmp_feature = feature.reshape(-1, feature.shape[-1]).detach().numpy()
        pca.fit(tmp_feature)
        pca_feature = pca.transform(tmp_feature)
        pca_feature = pca_feature[:, :3]
        for i in range(3): # min_max scaling
            pca_feature[:, i] = (pca_feature[:, i] - pca_feature[:, i].min()) / (pca_feature[:, i].max() - pca_feature[:, i].min())
        pca_feature = pca_feature.reshape(feature.shape[0], feature.shape[1], 3)
    else:
        pca_feature = feature
    if return_res:
        return pca_feature
    if mask is not None:
        pca_feature[~mask] = 0
    plt.imshow(pca_feature)
    plt.axis('off')
    plt.show()

def pc_camera_to_world(pc, extrinsic):
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    pc = (R @ pc.T).T + T
    return pc

def tanslation_point_cloud(depth_map, rgb_image, camera_intrinsic, cam2world_matrix, view=True, mask=None):
    if mask is None:
        mask = np.ones((rgb_image.shape[0], rgb_image.shape[1]))
    depth_map = depth_map.reshape(depth_map.shape[0], depth_map.shape[1])
    rows, cols = depth_map.shape[0], depth_map.shape[1]
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    z = depth_map
    x = (u - camera_intrinsic[0][2]) * z / camera_intrinsic[0][0]
    y = (v - camera_intrinsic[1][2]) * z / camera_intrinsic[1][1]
    points = np.dstack((x, y, z))
    per_point_xyz = points.reshape(-1, 3)
    line_masks = mask.reshape(-1)
    per_point_rgb = rgb_image.reshape(-1, 3)
    # view_point_cloud_parts(per_point_xyz, actor_seg)
    point_xyz = []
    point_rgb = []
    
    point_xyz = per_point_xyz[np.where(line_masks)]
    point_rgb = per_point_rgb[np.where(line_masks)]
    pcd_camera = np.array(point_xyz)
    point_rgb = np.array(point_rgb)
    Rtilt_rot = cam2world_matrix[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Rtilt_trl = cam2world_matrix[:3, 3]
    cam2_wolrd = np.eye(4)
    cam2_wolrd[:3, :3] = Rtilt_rot
    cam2_wolrd[:3, 3] = Rtilt_trl
    pcd_world = pc_camera_to_world(pcd_camera, cam2_wolrd)
    return pcd_world, point_rgb


def png_to_gif(png_folder, output_gif, num):

    frames = []
    for i in range(0, num, 5):
        path = os.path.join(png_folder, f'{i}.png')
        img = Image.open(path)
        img.load()  # 确保图像被加载
        frames.append(img)
    width, height = frames[0].size
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], loop=0, duration=200)

def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f'Folder {folder_path} has been deleted.')
    except Exception as e:
        pass
        # print(f'deleting Error: {e}')

def convert(x):
    """Recursively convert numpy arrays to lists."""
    if hasattr(x, "tolist"):  # Convert numpy arrays to lists
        return x.tolist()
    elif isinstance(x, dict):  # Recursively convert dicts
        return {key: convert(value) for key, value in x.items()}
    elif isinstance(x, list):  # Recursively convert lists
        return [convert(element) for element in x]
    else:
        return x


def load_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    pcd_array = np.hstack((np.array(pcd.points), np.array(pcd.colors)))
    return pcd_array

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def pcd2mp4(pcd_list, save_path, json_path, fps = 30):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=800, height=600)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 3
    opt.show_coordinate_frame = True

    param = o3d.io.read_pinhole_camera_parameters(json_path)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)

    pointcloud = o3d.geometry.PointCloud()

    to_reset = True

    img_list = []
    writer = imageio.get_writer(save_path, fps=fps)
    for pcd in pcd_list:
        time.sleep(0.02)
        pointcloud.points = o3d.utility.Vector3dVector(pcd[:,:3])
        pointcloud.colors = o3d.utility.Vector3dVector(pcd[:,3:])

        vis.update_geometry(pointcloud)
        vis.add_geometry(pointcloud)
        ctr.convert_from_pinhole_camera_parameters(param)

        if to_reset:
            vis.reset_view_point(True)
            to_reset = False
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer()
        img = (np.array(img) * 255).astype(np.uint8)
        writer.append_data(img)

    writer.close()

def save_pcd(save_path, pcd_file):
    ensure_dir(save_path)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd_file[:,:3])
    point_cloud.colors = o3d.utility.Vector3dVector(pcd_file[:,3:])
    o3d.io.write_point_cloud(save_path, point_cloud) 

def get_bounding_box_mask(mask):
    rows, cols = np.where(mask)
    if len(rows) > 0:
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        len_row, len_col = max_row - min_row, max_col - min_col
        padding = max(len_row, len_col) // 10

        min_row = max(0, min_row - padding)
        max_row = min(mask.shape[0], max_row + padding + 1)
        min_col = max(0, min_col - padding)
        max_col = min(mask.shape[1], max_col + padding + 1)

        region_mask = np.zeros_like(mask, dtype=bool)
        region_mask[min_row:max_row, min_col:max_col] = True
    else:
        region_mask = np.zeros_like(mask, dtype=bool)

    return region_mask

def extract_bounding_box_region(color, depth, bounding_box_mask, mask):
    rows, cols = np.where(bounding_box_mask)
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    region_image = color[min_row:max_row+1, min_col:max_col+1]
    region_mask = mask[min_row:max_row+1, min_col:max_col+1]
    region_depth = None
    if depth is not None:
        region_depth = depth[min_row:max_row+1, min_col:max_col+1]

    return region_image, region_depth, region_mask

def feature_to_rgb(feature_line, H, W):
    for i in range(3):
        feature_line[:, i] = (feature_line[:, i] - feature_line[:, i].min()) / \
                            (feature_line[:, i].max() - feature_line[:, i].min())
    feature_rgb = feature_line.reshape(H, W, -1)
    return feature_rgb

def get_dino_feature(image, transform_size=420, model=None):
    img, H, W = transform_np_image_to_torch(image, transform_size=transform_size) 
    res = model(img) # torch.Size([1, 384, 30, 30])
    feature = np.array(res.cpu().unsqueeze(0))
    new_order = (0, 1, 3, 4, 2) # torch.Size([1, 30, 30, 384])
    feature = np.transpose(feature, new_order)
    orig_shape_feature = transform_shape(torch.Tensor(np.transpose(feature[0], (0, 3, 1, 2))), H, W)
    orig_shape_feature_line = orig_shape_feature.reshape(-1, orig_shape_feature.shape[-1])
    return orig_shape_feature, orig_shape_feature_line

def fps(points, num_points=1024, use_cuda=True):

    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices


def find_mask_center(mask):
    mask = mask.squeeze(0)
    coords = torch.nonzero(mask, as_tuple=False)
    center = torch.mean(coords.float(), dim=0) 
    return center

def sort_masks(masks):
    centers = torch.stack([find_mask_center(mask) for mask in masks])
    sorted_indices = torch.argsort(centers[:, 1])
    sorted_masks = masks[sorted_indices]
    
    return sorted_masks