#! /usr/bin/env python3

r"""
Gaussian Process Network.
"""

from __future__ import annotations
import torch
from typing import Any
from botorch.models.model import Model
from botorch.models import FixedNoiseGP
from botorch import fit_gpytorch_model
from botorch.posteriors import Posterior
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


class GaussianProcessNetwork(Model):
    r""" """

    def __init__(
        self,
        train_X,
        train_Y,
        dag,
        active_input_indices,
        train_Yvar=None,
        node_GPs=None,
        normalization_constant_lower=None,
        normalization_constant_upper=None,
    ) -> None:
        r""" """
        self.train_X = train_X
        self.train_Y = train_Y
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.active_input_indices = active_input_indices
        self.train_Yvar = train_Yvar

        if node_GPs is not None:
            self.node_GPs = node_GPs
            self.normalization_constant_lower = normalization_constant_lower
            self.normalization_constant_upper = normalization_constant_upper
        else:
            self.node_GPs = [None for k in range(self.n_nodes)]
            self.node_mlls = [None for k in range(self.n_nodes)]
            self.normalization_constant_lower = [
                [None for j in range(len(self.dag.get_parent_nodes(k)))]
                for k in range(self.n_nodes)
            ]
            self.normalization_constant_upper = [
                [None for j in range(len(self.dag.get_parent_nodes(k)))]
                for k in range(self.n_nodes)
            ]

            for k in self.root_nodes:
                if self.active_input_indices is not None:
                    train_X_node_k = train_X[..., self.active_input_indices[k]]
                else:
                    train_X_node_k = train_X
                train_Y_node_k = train_Y[..., [k]]
                # self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))
                self.node_GPs[k] = FixedNoiseGP(
                    train_X=train_X_node_k,
                    train_Y=train_Y_node_k,
                    train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-4,
                    outcome_transform=Standardize(m=1),
                )
                self.node_mlls[k] = ExactMarginalLogLikelihood(
                    self.node_GPs[k].likelihood, self.node_GPs[k]
                )
                fit_gpytorch_model(self.node_mlls[k])

            for k in range(self.n_nodes):
                if self.node_GPs[k] is None:
                    aux = train_Y[..., self.dag.get_parent_nodes(k)].clone()
                    for j in range(len(self.dag.get_parent_nodes(k))):
                        self.normalization_constant_lower[k][j] = torch.min(aux[..., j])
                        self.normalization_constant_upper[k][j] = torch.max(aux[..., j])
                        aux[..., j] = (
                            aux[..., j] - self.normalization_constant_lower[k][j]
                        ) / (
                            self.normalization_constant_upper[k][j]
                            - self.normalization_constant_lower[k][j]
                        )
                    train_X_node_k = torch.cat(
                        [train_X[..., self.active_input_indices[k]], aux], -1
                    )
                    train_Y_node_k = train_Y[..., [k]]
                    aux_model = FixedNoiseGP(
                        train_X=train_X_node_k,
                        train_Y=train_Y_node_k,
                        train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-4,
                        outcome_transform=Standardize(m=1),
                    )
                    batch_shape = aux_model._aug_batch_shape
                    # self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))
                    # self.node_GPs[k] = FixedNoiseGP(train_X=train_X_node_k, train_Y=train_Y_node_k, train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-4, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))
                    self.node_GPs[k] = FixedNoiseGP(
                        train_X=train_X_node_k,
                        train_Y=train_Y_node_k,
                        train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-4,
                        outcome_transform=Standardize(m=1, batch_shape=torch.Size([])),
                    )
                    self.node_mlls[k] = ExactMarginalLogLikelihood(
                        self.node_GPs[k].likelihood, self.node_GPs[k]
                    )
                    fit_gpytorch_model(self.node_mlls[k])

    def posterior(
        self, X: Tensor, observation_noise=False
    ) -> MultivariateNormalNetwork:
        r"""Computes the posterior over model outputs at the provided points.
        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).
        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        return MultivariateNormalNetwork(
            self.node_GPs,
            self.dag,
            X,
            self.active_input_indices,
            self.normalization_constant_lower,
            self.normalization_constant_upper,
        )

    def forward(self, x: Tensor) -> MultivariateNormalNetwork:
        return MultivariateNormalNetwork(
            self.node_GPs,
            self.dag,
            x,
            self.active_input_indices,
            self.normalization_constant,
        )

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        r"""Condition the model on new observations.
        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n' x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, it is assumed that the missing batch dimensions are
                the same for all `Y`.
        Returns:
            A `Model` object of the same type, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).
        """
        fantasy_models = [None for k in range(self.n_nodes)]

        for k in self.root_nodes:
            if self.active_input_indices is not None:
                X_node_k = X[..., self.active_input_indices[k]]
            else:
                X_node_k = X
            Y_node_k = Y[..., [k]]
            fantasy_models[k] = self.node_GPs[k].condition_on_observations(
                X_node_k, Y_node_k, noise=torch.ones(Y_node_k.shape[1:]) * 1e-4
            )

        for k in range(self.n_nodes):
            if fantasy_models[k] is None:
                aux = Y[..., self.dag.get_parent_nodes(k)].clone()
                for j in range(len(self.dag.get_parent_nodes(k))):
                    aux[..., j] = (
                        aux[..., j] - self.normalization_constant_lower[k][j]
                    ) / (
                        self.normalization_constant_upper[k][j]
                        - self.normalization_constant_lower[k][j]
                    )
                aux_shape = [aux.shape[0]] + [1] * X[
                    ..., self.active_input_indices[k]
                ].ndim
                X_aux = (
                    X[..., self.active_input_indices[k]].unsqueeze(0).repeat(*aux_shape)
                )
                X_node_k = torch.cat([X_aux, aux], -1)
                Y_node_k = Y[..., [k]]
                fantasy_models[k] = self.node_GPs[k].condition_on_observations(
                    X_node_k, Y_node_k, noise=torch.ones(Y_node_k.shape[1:]) * 1e-4
                )

        return GaussianProcessNetwork(
            dag=self.dag,
            train_X=X,
            train_Y=Y,
            active_input_indices=self.active_input_indices,
            node_GPs=fantasy_models,
            normalization_constant_lower=self.normalization_constant_lower,
            normalization_constant_upper=self.normalization_constant_upper,
        )


class MultivariateNormalNetwork(Posterior):
    def __init__(
        self,
        node_GPs,
        dag,
        X,
        indices_X=None,
        normalization_constant_lower=None,
        normalization_constant_upper=None,
    ):
        self.node_GPs = node_GPs
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.X = X
        self.active_input_indices = indices_X
        self.normalization_constant_lower = normalization_constant_lower
        self.normalization_constant_upper = normalization_constant_upper

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return "cpu"

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return torch.double

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        shape = [self.X.shape[-2], self.n_nodes]
        shape = torch.Size(shape)
        return self.batch_shape + shape

    @property
    def batch_shape(self) -> torch.Size:
        """compute the batch shape of the GaussianProcessNetwork posterior."""
        gp_batch_shape = torch.broadcast_shapes(
            *[gp.batch_shape for gp in self.node_GPs]
        )
        X_batch_shape = self.X.shape[:-2]
        return torch.broadcast_shapes(gp_batch_shape, X_batch_shape)

    @property
    def base_sample_shape(self) -> torch.Size:
        """Compute the base sample shape of the GaussianProcessNetwork posterior."""
        return self.event_shape

    @property
    def _extended_shape(self, sample_shape: torch.Size) -> torch.Size:
        return sample_shape + self.base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        r"""The t-batch range.

        This is used in samplers to identify the t-batch component of the
        `base_sample_shape`. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        return (0, -2)

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        # print(sample_shape)
        # print(base_samples.shape)
        # print(self.X.shape)
        # nodes_samples = torch.empty(base_samples.shape)
        nodes_samples = torch.empty(sample_shape + self.event_shape)
        # print(nodes_samples.shape)
        # print(base_samples.shape)
        # print(e)
        nodes_samples = nodes_samples.to(self.device).to(self.dtype)
        nodes_samples_available = [False for k in range(self.n_nodes)]
        # batch_shape = base_samples.shape[:-2]
        # print(batch_shape)
        # print(self.X.shape[-2:])

        # if len(batch_shape) > 0:
        # self.X = torch.broadcast_to(self.X.unsqueeze(0), batch_shape + self.X.shape[-2:])

        for k in self.root_nodes:
            if self.active_input_indices is not None:
                X_node_k = self.X[..., self.active_input_indices[k]]
            else:
                X_node_k = self.X
            multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
            nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(
                sample_shape, base_samples[..., [k]]
            )[..., 0]
            nodes_samples_available[k] = True

        while not all(nodes_samples_available):
            for k in range(self.n_nodes):
                parent_nodes = self.dag.get_parent_nodes(k)
                if not nodes_samples_available[k] and all(
                    [nodes_samples_available[j] for j in parent_nodes]
                ):
                    parent_nodes_samples_normalized = nodes_samples[
                        ..., parent_nodes
                    ].clone()
                    for j in range(len(parent_nodes)):
                        parent_nodes_samples_normalized[..., j] = (
                            parent_nodes_samples_normalized[..., j]
                            - self.normalization_constant_lower[k][j]
                        ) / (
                            self.normalization_constant_upper[k][j]
                            - self.normalization_constant_lower[k][j]
                        )
                    X_node_k = self.X[..., self.active_input_indices[k]]
                    aux_shape = [sample_shape[0]] + [1] * X_node_k.ndim
                    X_node_k = X_node_k.unsqueeze(0).repeat(*aux_shape)
                    X_node_k = torch.cat(
                        [X_node_k, parent_nodes_samples_normalized], -1
                    )
                    multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
                    nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(
                        sample_shape=torch.Size([1]),
                        base_samples=base_samples[..., [k]].unsqueeze(dim=0),
                    )[0, ..., 0]
                    nodes_samples_available[k] = True
        return nodes_samples
