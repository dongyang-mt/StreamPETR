import os
from torch_musa.utils.simple_porting import SimplePorting

golden_path = './mmdetection3d/mmdet3d/ops/'
ops_list = ['spconv', 'iou3d', 'bev_pool', 'voxel', 'roiaware_pool3d', 'ball_query', 'knn', 'paconv', 'group_points', 'interpolate', 'furthest_point_sample', 'gather_points', 'bev_pool_v2']
c_paths = ['src', 'include']
for ops in ops_list:
    for c_path in c_paths:
        cuda_dir_path = os.path.join(golden_path, ops, c_path)
        if os.path.exists(cuda_dir_path):
            SimplePorting(cuda_dir_path=cuda_dir_path, 
                        mapping_rule={
                        "_CU_H_": "_MU_H_",
                        "_CUH": "_MUH",
                        "__NVCC__": "__MUSACC__",
                        "MMCV_WITH_CUDA": "MMCV_WITH_MUSA",
                        "AT_DISPATCH_FLOATING_TYPES_AND_HALF": "AT_DISPATCH_FLOATING_TYPES",
                        "#include <ATen/cuda/CUDAContext.h>": "#include \"torch_musa/csrc/aten/musa/MUSAContext.h\"",
                        "#include <c10/cuda/CUDAGuard.h>": "#include \"torch_musa/csrc/core/MUSAGuard.h\"",
                        "::cuda::": "::musa::",
                        "/cuda/": "/musa/",
                        ", CUDA,": ", PrivateUse1,",
                        ".cuh": ".muh",
                        ".is_cuda()": ".is_privateuseone()",
                        }
                        ).run()
