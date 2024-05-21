import jax.numpy as jnp
import numpy as np
import jax
import einops

def rand_sphere(outer_shape):
    ext = np.random.normal(size=outer_shape + (5,))
    return (ext / np.linalg.norm(ext, axis=-1, keepdims=True))[...,-3:]


def safe_norm(x, axis, keepdims=False, eps=0.0):
    is_zero = jnp.all(jnp.isclose(x,0.), axis=axis, keepdims=True)
    # temporarily swap x with ones if is_zero, then swap back
    x = jnp.where(is_zero, jnp.ones_like(x), x)
    n = jnp.linalg.norm(x, axis=axis, keepdims=keepdims)
    n = jnp.where(is_zero if keepdims else jnp.squeeze(is_zero, -1), 0., n)
    return n.clip(eps)

# quaternion operations
def normalize(vec, eps=1e-8):
    # return vec/(safe_norm(vec, axis=-1, keepdims=True, eps=eps) + 1e-8)
    return vec/safe_norm(vec, axis=-1, keepdims=True, eps=eps)

def quw2wu(quw):
    return jnp.concatenate([quw[...,-1:], quw[...,:3]], axis=-1)

def qrand(outer_shape, jkey=None):
    if jkey is None:
        return qrand_np(outer_shape)
    else:
        return normalize(jax.random.normal(jkey, outer_shape + (4,)))

def qrand_np(outer_shape):
    q = np.random.normal(size=outer_shape+(4,))
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q

def line2q(zaxis, yaxis=jnp.array([1,0,0])):
    Rm = line2Rm(zaxis, yaxis)
    return Rm2q(Rm)

def qmulti(q1, q2):
    b,c,d,a = jnp.split(q1, 4, axis=-1)
    f,g,h,e = jnp.split(q2, 4, axis=-1)
    w,x,y,z = a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g, a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e
    return jnp.concatenate([x,y,z,w], axis=-1)

def qinv(q):
    x,y,z,w = jnp.split(q, 4, axis=-1)
    return jnp.concatenate([-x,-y,-z,w], axis=-1)

def qlog(q):
    alpha = jnp.arccos(q[...,3:])
    sinalpha = jnp.sin(alpha)
    abssinalpha = jnp.maximum(jnp.abs(sinalpha), 1e-6)
    n = q[...,:3]/(abssinalpha*jnp.sign(sinalpha))
    return jnp.where(jnp.abs(q[...,3:])<1-1e-6, n*alpha, jnp.zeros_like(n))

def q2aa(q):
    return 2*qlog(q)

def qLog(q):
    return qvee(qlog(q))

def qvee(phi):
    return 2*phi[...,:-1]

def qhat(w):
    return jnp.concatenate([w*0.5, jnp.zeros_like(w[...,0:1])], axis=-1)

def aa2q(aa):
    return qexp(aa*0.5)

def q2R(q):
    i,j,k,r = jnp.split(q, 4, axis=-1)
    R1 = jnp.concatenate([1-2*(j**2+k**2), 2*(i*j-k*r), 2*(i*k+j*r)], axis=-1)
    R2 = jnp.concatenate([2*(i*j+k*r), 1-2*(i**2+k**2), 2*(j*k-i*r)], axis=-1)
    R3 = jnp.concatenate([2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i**2+j**2)], axis=-1)
    return jnp.stack([R1,R2,R3], axis=-2)

def qexp(logq):
    if isinstance(logq, np.ndarray):
        alpha = np.linalg.norm(logq[...,:3], axis=-1, keepdims=True)
        alpha = np.maximum(alpha, 1e-6)
        return np.concatenate([logq[...,:3]/alpha*np.sin(alpha), np.cos(alpha)], axis=-1)
    else:
        alpha = safe_norm(logq[...,:3], axis=-1, keepdims=True)
        alpha = jnp.maximum(alpha, 1e-6)
        return jnp.concatenate([logq[...,:3]/alpha*jnp.sin(alpha), jnp.cos(alpha)], axis=-1)

def qExp(w):
    return qexp(qhat(w))

def qaction(quat, pos):
    return qmulti(qmulti(quat, jnp.concatenate([pos, jnp.zeros_like(pos[...,:1])], axis=-1)), qinv(quat))[...,:3]

def qnoise(quat, scale=np.pi*10/180):
    lq = np.random.normal(scale=scale, size=quat[...,:3].shape)
    return qmulti(quat, qexp(lq))

# posquat operations
def pq_inv(pos, quat):
    quat_inv = qinv(quat)
    return -qaction(quat_inv, pos), quat_inv

def pq_action(translate, rotate, pnt):
    return qaction(rotate, pnt) + translate

def pq_multi(pos1, quat1, pos2, quat2):
    return qaction(quat1, pos2)+pos1, qmulti(quat1, quat2)

def pq2H(pos, quat):
    R = q2R(quat)
    return H_from_Rpos(R, pos)

# homogineous transforms
def H_from_Rpos(R, pos):
    H = jnp.zeros(pos.shape[:-1] + (4,4))
    H = H.at[...,-1,-1].set(1)
    H = H.at[...,:3,:3].set(R)
    H = H.at[...,:3,3].set(pos)
    return H

def H_inv(H):
    R = H[...,:3,:3]
    p = H[...,:3, 3:]
    return H_from_Rpos(T(R), (-T(R)@p)[...,0])

def H2pq(H, concat=False):
    R = H[...,:3,:3]
    p = H[...,:3, 3]
    if concat:
        return jnp.concatenate([p, Rm2q(R)], axis=-1)
    else:
        return p, Rm2q(R)

# Rm util
def Rm_inv(Rm):
    return T(Rm)

def line2Rm(zaxis, yaxis=jnp.array([1,0,0])):
    zaxis = normalize(zaxis + jnp.array([0,1e-6,0]))
    xaxis = jnp.cross(yaxis, zaxis)
    xaxis = normalize(xaxis)
    yaxis = jnp.cross(zaxis, xaxis)
    Rm = jnp.stack([xaxis, yaxis, zaxis], axis=-1)
    return Rm

def line2Rm_np(zaxis, yaxis=np.array([1,0,0])):
    zaxis = (zaxis + jnp.array([0,1e-6,0]))
    zaxis = zaxis/np.linalg.norm(zaxis, axis=-1, keepdims=True)
    xaxis = np.cross(yaxis, zaxis)
    xaxis = xaxis/np.linalg.norm(xaxis, axis=-1, keepdims=True)
    yaxis = np.cross(zaxis, xaxis)
    Rm = np.stack([xaxis, yaxis, zaxis], axis=-1)
    return Rm

def Rm2q(Rm):
    Rm = einops.rearrange(Rm, '... i j -> ... j i')
    con1 = (Rm[...,2,2] < 0) & (Rm[...,0,0] > Rm[...,1,1])
    con2 = (Rm[...,2,2] < 0) & (Rm[...,0,0] <= Rm[...,1,1])
    con3 = (Rm[...,2,2] >= 0) & (Rm[...,0,0] < -Rm[...,1,1])
    con4 = (Rm[...,2,2] >= 0) & (Rm[...,0,0] >= -Rm[...,1,1]) 

    t1 = 1 + Rm[...,0,0] - Rm[...,1,1] - Rm[...,2,2]
    t2 = 1 - Rm[...,0,0] + Rm[...,1,1] - Rm[...,2,2]
    t3 = 1 - Rm[...,0,0] - Rm[...,1,1] + Rm[...,2,2]
    t4 = 1 + Rm[...,0,0] + Rm[...,1,1] + Rm[...,2,2]

    q1 = jnp.stack([t1, Rm[...,0,1]+Rm[...,1,0], Rm[...,2,0]+Rm[...,0,2], Rm[...,1,2]-Rm[...,2,1]], axis=-1) / jnp.sqrt(t1.clip(1e-7))[...,None]
    q2 = jnp.stack([Rm[...,0,1]+Rm[...,1,0], t2, Rm[...,1,2]+Rm[...,2,1], Rm[...,2,0]-Rm[...,0,2]], axis=-1) / jnp.sqrt(t2.clip(1e-7))[...,None]
    q3 = jnp.stack([Rm[...,2,0]+Rm[...,0,2], Rm[...,1,2]+Rm[...,2,1], t3, Rm[...,0,1]-Rm[...,1,0]], axis=-1) / jnp.sqrt(t3.clip(1e-7))[...,None]
    q4 = jnp.stack([Rm[...,1,2]-Rm[...,2,1], Rm[...,2,0]-Rm[...,0,2], Rm[...,0,1]-Rm[...,1,0], t4], axis=-1) / jnp.sqrt(t4.clip(1e-7))[...,None]
 
    q = jnp.zeros(Rm.shape[:-2]+(4,))
    q = jnp.where(con1[...,None], q1, q)
    q = jnp.where(con2[...,None], q2, q)
    q = jnp.where(con3[...,None], q3, q)
    q = jnp.where(con4[...,None], q4, q)
    q *= 0.5

    return q

def pRm_inv(pos, Rm):
    return (-T(Rm)@pos[...,None,:])[...,0], T(Rm)

def pRm_action(pos, Rm, x):
    return (Rm @ x[...,None,:])[...,0] + pos

# 6d utils
def R6d2Rm(x, gram_schmidt=False):
    xv, yv = x[...,:3], x[...,3:]
    xv = normalize(xv)
    if gram_schmidt:
        yv = normalize(yv - jnp.einsum('...i,...i',yv,xv)[...,None]*xv)
        zv = jnp.cross(xv, yv)
    else:
        zv = jnp.cross(xv, yv)
        zv = normalize(zv)
        yv = jnp.cross(zv, xv)
    return jnp.stack([xv,yv,zv], -1)

# 9d utils
def R9d2Rm(x):
    xm = einops.rearrange(x, '... (t i) -> ... t i', t=3)
    u, s, vt = jnp.linalg.svd(xm)
    # vt = einops.rearrange(v, '... i j -> ... j i')
    det = jnp.linalg.det(jnp.matmul(u,vt))
    vtn = jnp.concatenate([vt[...,:2,:], vt[...,2:,:]*det[...,None,None]], axis=-2)
    return jnp.matmul(u,vtn)


# general
def T(mat):
    return einops.rearrange(mat, '... i j -> ... j i')


# euler angle
def Rm2ZYZeuler(Rm):
    sy = jnp.sqrt(Rm[...,0,2]**2+Rm[...,1,2]**2)
    v1 = jnp.arctan2(Rm[...,1,2], Rm[...,0,2])
    v2 = jnp.arctan2(sy, Rm[...,2,2])
    v3 = jnp.arctan2(Rm[...,2,1], -Rm[...,2,0])

    v1n = jnp.arctan2(-Rm[...,0,1], Rm[...,1,1])
    v1 = jnp.where(sy < 1e-6, v1n, v1)
    v3 = jnp.where(sy < 1e-6, jnp.zeros_like(v1), v3)

    return jnp.stack([v1,v2,v3],-1)

def Rm2YXYeuler(Rm):
    sy = jnp.sqrt(jnp.sqrt(Rm[...,0,1]**2+Rm[...,2,1]**2))
    v1 = jnp.arctan2(Rm[...,0,1], Rm[...,2,1])
    v2 = jnp.arctan2(sy, Rm[...,1,1])
    v3 = jnp.arctan2(Rm[...,1,0], -Rm[...,1,2])

    v1n = jnp.arctan2(-Rm[...,2,0], Rm[...,0,0])
    v1 = jnp.where(sy < 1e-6, v1n, v1)
    v3 = jnp.where(sy < 1e-6, jnp.zeros_like(v1), v3)

    return jnp.stack([v1,v2,v3],-1)

def YXYeuler2Rm(YXYeuler):
    c1,c2,c3 = jnp.split(jnp.cos(YXYeuler), 3, -1)
    s1,s2,s3 = jnp.split(jnp.sin(YXYeuler), 3, -1)
    return jnp.stack([jnp.concatenate([c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],-1),
            jnp.concatenate([s2*s3, c2, -c3*s2],-1),
            jnp.concatenate([-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3],-1)], -2)

def wigner_D_order1_from_Rm(Rm):
    r1,r2,r3 = jnp.split(Rm,3,-2)
    r11,r12,r13 = jnp.split(r1,3,-1)
    r21,r22,r23 = jnp.split(r2,3,-1)
    r31,r32,r33 = jnp.split(r3,3,-1)

    return jnp.concatenate([jnp.c_[r22, r23, r21],
                jnp.c_[r32, r33, r31],
                jnp.c_[r12, r13, r11]], axis=-2)

def q2ZYZeuler(q):
    return Rm2ZYZeuler(q2R(q))

def q2XYZeuler(q):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = jnp.split(q, 4, -1)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = jnp.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = jnp.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = jnp.arctan2(t3, t4)
    
    return jnp.concatenate([roll_x, pitch_y, yaw_z], -1) # in radians


# widger D matrix
Jd = [None,None,None,None,None]
Jd[1] = jnp.array([[ 0., -1.,  0.],
        [-1.,  0.,  0.],
        [ 0.,  0.,  1.]])
Jd[2] = jnp.array([[ 0.       ,  0.       ,  0.       , -1.       ,  0.       ],
       [ 0.       ,  1.       ,  0.       ,  0.       ,  0.       ],
       [ 0.       ,  0.       , -0.5      ,  0.       , -0.8660254],
       [-1.       ,  0.       ,  0.       ,  0.       ,  0.       ],
       [ 0.       ,  0.       , -0.8660254,  0.       ,  0.5      ]])
Jd[3] = jnp.array([[ 0.        ,  0.        ,  0.        ,  0.79056942,  0.        , -0.61237244,  0.        ],
       [ 0.        ,  1.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.61237244,  0.        , 0.79056942,  0.        ],
       [ 0.79056942,  0.        ,  0.61237244,  0.        ,  0.        , 0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        , -0.25      , 0.        , -0.96824584],
       [-0.61237244,  0.        ,  0.79056942,  0.        ,  0.        , 0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        , -0.96824584, 0.        ,  0.25      ]])
   
def wigner_d_matrix(degree, ZYZeuler):
    '''
    here, alpha, beta, gamma: alpha, beta, gamma = sciR.from_quat(quat).as_euler('ZYZ')
    ZYZ euler with relative Rz@Ry@Rz
    Note that when degree is 1 wigner_d matrix is not equal to rotation matrix
    The equality comes from sciR.from_quat(q).as_matrix() = wigner_d_matrix(1, *sciR.from_quat(q).as_euler('YXY'))
    '''
    """Create wigner D matrices for batch of ZYZ Euler anglers for degree l."""
    if degree==0:
        return jnp.array([[1.0]])
    if degree==1:
        return YXYeuler2Rm(ZYZeuler)
    origin_outer_shape = ZYZeuler.shape[:-1]
    ZYZeuler = ZYZeuler.reshape((-1,3))
    alpha, beta, gamma = jnp.split(ZYZeuler,3,-1)
    J = Jd[degree]
    x_a = z_rot_mat(alpha, degree)
    x_b = z_rot_mat(beta, degree)
    x_c = z_rot_mat(gamma, degree)
    res = x_a @ J @ x_b @ J @ x_c
    return res.reshape(origin_outer_shape+res.shape[-2:])

def wigner_d_from_quat(degree, quat):
    return wigner_d_from_RmV2(degree, q2R(quat))
    # if degree==1:
    #     return wigner_D_order1_from_Rm(q2R(quat))
    # return wigner_d_matrix(degree, q2ZYZeuler(quat))

def wigner_d_from_Rm(degree, Rm):
    if degree==1:
        return wigner_D_order1_from_Rm(Rm)
    return wigner_d_matrix(degree, Rm2ZYZeuler(Rm))

def wigner_d_from_RmV2(degree, Rm):
    # assert degree <= 3
    if degree==0:
        return 1
    if degree==1:
        return wigner_D_order1_from_Rm(Rm)
    if degree > 3:
        return wigner_d_from_Rm(degree, Rm)
    Rm_flat = einops.rearrange(Rm, '... i j -> ... (i j)')
    Rm_concat = einops.rearrange((Rm_flat[...,None]*Rm_flat[...,None,:]), '... i j -> ... (i j)')
    if degree==3:
        Rm_concat = einops.rearrange((Rm_concat[...,None]*Rm_flat[...,None,:]), '... i j -> ... (i j)')
    
    return einops.rearrange(jnp.einsum('...i,...ik', Rm_concat, WDCOEF[degree]), '... (r i)-> ... r i', r=2*degree+1)



def z_rot_mat(angle, l):
    '''
    angle : (... 1)
    '''
    outer_shape = angle.shape[:-1]
    order = 2*l+1
    m = jnp.zeros(outer_shape + (order, order))
    inds = jnp.arange(0, order)
    reversed_inds = jnp.arange(2*l, -1, -1)
    frequencies = jnp.arange(l, -l -1, -1)

    m = m.at[..., inds, reversed_inds].set(jnp.sin(frequencies * angle))
    m = m.at[..., inds, inds].set(jnp.cos(frequencies * angle))
    return m


def x_to_alpha_beta(x):
    '''
    Convert point (x, y, z) on the sphere into (alpha, beta)
    '''
    # x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    x = normalize(x)
    beta = jnp.arccos(x[...,2])
    alpha = jnp.arctan2(x[...,1], x[...,0])
    return alpha, beta

def sh_via_wigner_d(l, pnt):
    a, b = x_to_alpha_beta(pnt)
    return wigner_d_matrix(l, jnp.stack([a, b, jnp.zeros_like(a)], -1))[...,:,l]


import os
import pickle
if not os.path.exists('Wigner_D_coef.pkl'):
    WDCOEF = [None,None,None,None]
    ns_ = 100000
    Rmin = q2R(qrand((ns_,)))
    Rmin = np.array(Rmin).astype(np.float64)
    Rmin_flat = Rmin.reshape(-1,9)

    # order 2
    y_ = np.array(wigner_d_from_Rm(2,Rmin).reshape((ns_,-1))).astype(np.float64)
    Rmin_concat = (Rmin_flat[...,None]*Rmin_flat[...,None,:]).reshape((ns_,-1))
    WDCOEF[2] = np.linalg.pinv(Rmin_concat)@y_
    WDCOEF[2] = np.where(np.abs(WDCOEF[2])<1e-5, 0, WDCOEF[2])
    print(np.max(np.abs(Rmin_concat@WDCOEF[2]-y_)))

    #order 3
    Rmin_concat = (Rmin_concat[...,None]*Rmin_flat[...,None,:]).reshape((ns_,-1)).astype(np.float64)
    y_ = np.array(wigner_d_from_Rm(3,Rmin).reshape((ns_,-1))).astype(np.float64)
    WDCOEF[3] = np.linalg.pinv(Rmin_concat)@y_
    WDCOEF[3] = np.where(np.abs(WDCOEF[3])<1e-5, 0, WDCOEF[3])

    print(np.max(np.abs(Rmin_concat@WDCOEF[3]-y_)))
    with open('Wigner_D_coef.pkl', 'wb') as f:
        pickle.dump(WDCOEF, f)
    del Rmin, y_, Rmin_concat
else:
    with open('Wigner_D_coef.pkl', 'rb') as f:
        WDCOEF = pickle.load(f)
