# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

"""
Implements a joint lower limit map.
"""

import os
import torch
from fabrics_sim.taskmaps.maps_base import BaseMap

class LowerJointLimitMap(BaseMap):
    def __init__(self, lower_joint_limits, batch_size, device, joint_range=None):
        """
        Initializes the lower joint limit task map.
        -----------------------------
        @param lower_joint_limits: a list of joint limits
        @param joint_range: optional list of (upper - lower) per joint. If given, the
                            clearance is normalised by it (ADEPT App. B.4), x in [0, 1].
        @param device: type str that sets the cuda device for the fabric
        """
        super().__init__(device)

        self.lower_joint_limits = lower_joint_limits
        self.lower_joint_limits_batch = None
        self.batch_size = batch_size
        self.joint_range = joint_range

        self.init_limits()
    
    def init_limits(self):
        if (self.lower_joint_limits_batch is None):
            num_joints = len(self.lower_joint_limits)
            self.lower_joint_limits_batch =\
                torch.zeros(self.batch_size, num_joints, device=self.device)
            with torch.no_grad():
                for i in range(num_joints):
                    self.lower_joint_limits_batch[:,i] = self.lower_joint_limits[i]
        
                # Create the Jacobian
                dim = num_joints
                single_jacobian = torch.eye(dim, dim, device=self.device)
                single_jacobian = single_jacobian.reshape((1, dim, dim))
                self.jacobian_diagonal = torch.ones(dim, device=self.device)
                if self.joint_range is not None:
                    self.jacobian_diagonal /= torch.tensor(self.joint_range, device=self.device)
                    single_jacobian = torch.diag(self.jacobian_diagonal).reshape((1, dim, dim))
                self.jacobian = single_jacobian.repeat(self.batch_size, 1, 1)

    def forward_position(self, q, features):
        x = (q - self.lower_joint_limits_batch) * self.jacobian_diagonal

        return (x, self.jacobian)
