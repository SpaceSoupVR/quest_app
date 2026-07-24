
pub use avatar_ik::{
    body_root_transform, height_calibrated_scale, load_rig_config, HandCurl, LocalPose,
    RemotePlayerState, RigConfig, Transform,
};

use space_soup::renderer::mesh::GltfSkin;

pub fn skeleton_data_from_skin(skin: &GltfSkin) -> avatar_ik::SkeletonData {
    avatar_ik::SkeletonData {
        joint_names: skin.joint_names.clone(),
        joint_parents: skin.joint_parents.clone(),
        joint_local_bind: skin.joint_local_bind.clone(),
        inv_bind_mats: skin.inv_bind_mats.clone(),
    }
}

