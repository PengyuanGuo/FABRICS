# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

import os

import torch
import yaml
import time

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm

class JointLimitRepulsion(BaseFabricTerm):
    """
    Implements a fabric joint limit repulsion term.
    """
    def __init__(self, is_forcing_policy, params, device, graph_capturable):
        """
        Constructor.
        -----------------------------
        @param is_forcing_policy: indicates whether the acceleration policy
                                  will be forcing (as opposed to geometric).
        """
        super().__init__(is_forcing_policy, params, device, graph_capturable=graph_capturable)

        self._kEpsilon = 1e-6
        self.params['metric_scalar'] = torch.tensor(self.params['metric_scalar'], device=device)
        self.params['max_metric'] = torch.tensor(self.params['max_metric'], device=device)
        self._min_x_delta = self.compute_min_x_delta()

        self.ones_like_x = None

    def compute_min_x_delta(self):
        return torch.tensor((self.params['metric_scalar'] / self.params['max_metric'])**0.5,
                            device=self.device)

    def metric_eval(self, x, xd, features):
        """
        Evaluate the metric for this attractor term.
        -----------------------------
        @param x: position
        @param xd: velocity
        @param features: dictionary of features (inputs) to pass to this term.
        @return metric: policy metric
        """
        if self.metric is None:
            self.metric = torch.zeros(x.shape[0], x.shape[1], x.shape[1], requires_grad=False,
                                      device=self.device)
            self.force = torch.zeros(x.shape[0], x.shape[1], requires_grad=False,
                                      device=self.device)
            self.ones_like_x = torch.ones_like(x)
        
        if self.graph_capturable:
            self.metric.zero_().detach_()
            self.force.zero_().detach_()
        else:
            self.metric = torch.zeros_like(self.metric)
            self.force = torch.zeros_like(self.force)

        x_delta = torch.maximum(self._min_x_delta, x - self.params['metric_exploder_offset'])

        if self.params['velocity_gate']:
            # Trigger set to slightly positive position and velocity which helps with numerical
            # issues of this gate turning on/off with large dt. The effect of that is that you can
            # creep/oscillate through the joint limit.
            activation_trigger = torch.logical_or(xd < self.params['breakaway_velocity'],
                                                  x < self.params['breakaway_distance'])
            if self.graph_capturable:
                self.metric.copy_(torch.diag_embed(activation_trigger * self.params['metric_scalar'] / x_delta ** 2))
            else:
                self.metric =\
                    torch.diag_embed(activation_trigger * self.params['metric_scalar'] / x_delta ** 2)
        else:
            if self.graph_capturable:
                self.metric.copy_(torch.diag_embed(self.params['metric_scalar'] / x_delta ** 2))
            else:
                self.metric = torch.diag_embed(self.params['metric_scalar'] / x_delta ** 2)

    def force_eval(self, x, xd, features):
        """
        Evaluate the force for this repulsion term.
        -----------------------------
        @param x: position
        @param xd: velocity
        @param features: features (inputs) to pass to this term.
        @return force: batch bxn tensor of policy forces
        """

        if not self.is_forcing_policy:
            vel_squared = torch.sum(xd*xd, dim=1).unsqueeze(1)
            xdd = vel_squared * self.params['soft_relu_gain'] * self.ones_like_x
        else:
            # If velocity is negative (motion towards limit), then engage damping
            damping_gain = (xd <= 0) * self.params['damping_gain']
            xdd = self.params['soft_relu_gain'] * self.ones_like_x - damping_gain  * xd
        
        # Convert to force.
        if self.graph_capturable:
            self.force.copy_(-torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2))
        else:
            self.force = -torch.bmm(self.metric, xdd.unsqueeze(2)).squeeze(2)



class JointLimitBarrier(BaseFabricTerm):
    """
    Joint-limit barrier of ADEPT (arXiv 2608.19182) App. B.4, eqs. (19)-(22). Meant for the
    range-normalised clearance taskmaps x = (q_upper - q) / range, (q - q_lower) / range, so
    offsets are fractions of the joint range. Differences to JointLimitRepulsion: the
    bang-bang velocity gate is replaced by g = 0.5 (tanh(-gate_sharpness (xd - gate_offset)) + 1)
    on both metric and repulsion, and the metric stays diagonal (no dense bmm).

        M = diag(g k_b / max(x - delta_J, eps)^2),  eps = sqrt(k_b / max_metric)
        forcing:   xdd = g k_r - b_J 1{xd < 0} xd
        geometric: xdd = |xd|^2 k_r g

    Params: metric_scalar (k_b), metric_exploder_offset (delta_J), max_metric, soft_relu_gain
    (k_r), damping_gain (b_J), gate_sharpness (sigma_g), gate_offset (delta_g).

    Optional position floor on the gate (not in ADEPT): g = max(g, 0.5 (tanh(-hold_sharpness
    (x - hold_offset)) + 1)). With gate_offset < 0 the barrier is off at rest, so targets at
    the limit are reachable, and the floor still turns it fully on within the last
    hold_offset of the range when a slow push (e.g. body repulsion) creeps toward the limit.
    """
    def __init__(self, is_forcing_policy, params, device, graph_capturable):
        super().__init__(is_forcing_policy, params, device, graph_capturable=graph_capturable)
        self._eps = (self.params['metric_scalar'] / self.params['max_metric']) ** 0.5
        self._gate = None

    def metric_eval(self, x, xd, features):
        if self.metric is None:
            self.metric = torch.zeros(x.shape[0], x.shape[1], x.shape[1], requires_grad=False,
                                      device=self.device)
            self.force = torch.zeros(x.shape[0], x.shape[1], requires_grad=False,
                                     device=self.device)
            self._gate = torch.zeros_like(x)
            self._metric_diag = torch.zeros_like(x)

        gate = 0.5 * (torch.tanh(-self.params['gate_sharpness'] *
                                 (xd - self.params['gate_offset'])) + 1.)
        if 'hold_offset' in self.params:
            gate = torch.maximum(gate, 0.5 * (torch.tanh(-self.params['hold_sharpness'] *
                                 (x - self.params['hold_offset'])) + 1.))
        x_delta = torch.clamp(x - self.params['metric_exploder_offset'], min=self._eps)
        metric_diag = gate * self.params['metric_scalar'] / x_delta ** 2
        if self.graph_capturable:
            self._gate.copy_(gate)
            self._metric_diag.copy_(metric_diag)
            self.metric.copy_(torch.diag_embed(metric_diag))
        else:
            self._gate = gate
            self._metric_diag = metric_diag
            self.metric = torch.diag_embed(metric_diag)

    def force_eval(self, x, xd, features):
        if self.is_forcing_policy:
            xdd = self.params['soft_relu_gain'] * self._gate -\
                  (xd < 0) * self.params['damping_gain'] * xd
        else:
            xdd = torch.sum(xd * xd, dim=1, keepdim=True) * self.params['soft_relu_gain'] * self._gate
        if self.graph_capturable:
            self.force.copy_(-self._metric_diag * xdd)
        else:
            self.force = -self._metric_diag * xdd
