"""旋转表示转换（6D / 矩阵 / 四元数 / 轴角），供 272 维解析与重定向复用。"""

from __future__ import annotations

import numpy as np


def rotation_6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """6D 旋转（前两行）→ 旋转矩阵，第三行由正交化叉积得到。"""
    arr = np.asarray(rot6d, dtype=np.float64)
    if arr.shape[-1] != 6:
        raise ValueError(f"rot6d last dim must be 6, got {arr.shape}")
    a1, a2 = arr[..., :3], arr[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-9, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-9, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵 → wxyz 四元数（数值稳定分支）。"""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must be (...,3,3), got {m.shape}")
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    trace = m00 + m11 + m22
    quat = np.zeros(m.shape[:-2] + (4,), dtype=np.float64)
    cond0 = trace > 0.0
    s0 = np.sqrt(np.clip(trace + 1.0, 1e-12, None)) * 2.0
    quat[..., 0] = np.where(cond0, 0.25 * s0, quat[..., 0])
    quat[..., 1] = np.where(cond0, (m21 - m12) / s0, quat[..., 1])
    quat[..., 2] = np.where(cond0, (m02 - m20) / s0, quat[..., 2])
    quat[..., 3] = np.where(cond0, (m10 - m01) / s0, quat[..., 3])
    cond1 = np.logical_and(~cond0, m00 > m11)
    cond1 = np.logical_and(cond1, m00 > m22)
    s1 = np.sqrt(np.clip(1.0 + m00 - m11 - m22, 1e-12, None)) * 2.0
    quat[..., 0] = np.where(cond1, (m21 - m12) / s1, quat[..., 0])
    quat[..., 1] = np.where(cond1, 0.25 * s1, quat[..., 1])
    quat[..., 2] = np.where(cond1, (m01 + m10) / s1, quat[..., 2])
    quat[..., 3] = np.where(cond1, (m02 + m20) / s1, quat[..., 3])
    cond2 = np.logical_and(~cond0, ~cond1)
    cond2 = np.logical_and(cond2, m11 > m22)
    s2 = np.sqrt(np.clip(1.0 + m11 - m00 - m22, 1e-12, None)) * 2.0
    quat[..., 0] = np.where(cond2, (m02 - m20) / s2, quat[..., 0])
    quat[..., 1] = np.where(cond2, (m01 + m10) / s2, quat[..., 1])
    quat[..., 2] = np.where(cond2, 0.25 * s2, quat[..., 2])
    quat[..., 3] = np.where(cond2, (m12 + m21) / s2, quat[..., 3])
    cond3 = np.logical_and(~cond0, ~cond1)
    cond3 = np.logical_and(cond3, ~cond2)
    s3 = np.sqrt(np.clip(1.0 + m22 - m00 - m11, 1e-12, None)) * 2.0
    quat[..., 0] = np.where(cond3, (m10 - m01) / s3, quat[..., 0])
    quat[..., 1] = np.where(cond3, (m02 + m20) / s3, quat[..., 1])
    quat[..., 2] = np.where(cond3, (m12 + m21) / s3, quat[..., 2])
    quat[..., 3] = np.where(cond3, 0.25 * s3, quat[..., 3])
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.clip(norm, 1e-9, None)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """wxyz 四元数 → 旋转矩阵。"""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quat last dim must be 4, got {q.shape}")
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9, None)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def yaw_quat(yaw: float) -> np.ndarray:
    """绕 z 轴 yaw 的 wxyz 四元数。"""
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 轴角（rad）。"""
    m = np.asarray(matrix, dtype=np.float64)
    quat = matrix_to_quat_wxyz(m)
    return quat_to_axis_angle(quat)


def quat_to_axis_angle(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz 四元数 → 轴角（rad）。"""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9, None)
    w = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    axis = q[..., 1:4]
    sin_half = np.sqrt(np.clip(1.0 - w * w, 1e-12, None))
    axis = axis / np.clip(sin_half[..., None], 1e-9, None)
    return axis * angle[..., None]


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """四元数球面插值（t 形状 (T,)）。"""
    a, b = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = a[None, :] + np.asarray(t)[:, None] * (b - a)[None, :]
    else:
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        w0 = np.sin((1.0 - np.asarray(t)) * theta) / sin_theta
        w1 = np.sin(np.asarray(t) * theta) / sin_theta
        out = w0[:, None] * a[None, :] + w1[:, None] * b[None, :]
    return out / np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-9, None)
