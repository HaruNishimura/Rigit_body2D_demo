import pygame
import sys
import random
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import math
import csv

@dataclass(frozen=False)
class PhysicsSettings:
    """シミュレーション全体の設定を一括管理するクラス"""
    #世界の大きさを定義
    world_width:int = 800
    world_height:int = 600
    baumgarte: float = 0.1
    slop: float = 0.01
    #4分木の深さ
    morton_depth:int = 5
    # 静止時の微振動を防ぐための反発速度の閾値
    # 相対速度がこの値より小さい場合、反発係数を0とみなして跳ね返りを無効化します。
    # 目安: 重力加速度 * タイムステップ よりも少し大きい値 (例: 980 * 1/60s ≈ 16 なので 5~20 くらいが安全)
    restitution_threshold:float = 2.0
    #PGSソルバ(速度解決処理)の反復回数
    PGS_iterations:int = 1
    #サブステップ数
    sub_steps:int = 2
    #速度の丸め込み閾値
    velocity_threshold:float = 0.1
    #角速度の丸め込み閾値
    angular_threshold:float = 0.01
    #重力加速度
    gravity:tuple[float,float] = (0.0, 980.0)
    
class Transform2D:
    def __init__(self, pos=(0,0), angle=0.0, scale=(1.0, 1.0)):
        self._pos = pygame.math.Vector2(0, 0)
        self._angle = 0.0
        self._scale = pygame.math.Vector2(1, 1)
        
        self._matrix = np.identity(3)
        self._inv_matrix = np.identity(3)
        self._local_vertices = None
        self._world_vertices_cache:list[pygame.math.Vector2] = None
        self._is_dirty = True

        # 値変化を検知するオブザーバ
        self.on_angle_changed = []
        self.on_scale_changed = [] 

        self.pos = pos
        self.angle = angle
        self.scale = scale
    
    @property
    def pos(self):
        '''
        セッターを通過するため、Vector2のメゾット( vec.x = )や( vec.update(value) )などで書き換えないこと
        '''
        return self._pos
    @pos.setter
    def pos(self, value):
        self._pos.update(value)
        self._is_dirty = True
        self._world_vertices_cache = None
    @property
    def angle(self):
        '''
        セッターを通過するため、Vector2のメゾット( vec.x = )や( vec.update(value) )などで書き換えないこと
        '''
        return self._angle
    @angle.setter
    def angle(self, value):
        self._angle = value
        self._is_dirty = True
        self._world_vertices_cache = None
        #角度変化を検知
        for callback in self.on_angle_changed:
            callback()
    @property
    def scale(self):
        return self._scale
    @scale.setter
    def scale(self, value):
        self._scale.update(value)
        self._world_vertices_cache = None
        #スケールが変化した時、登録された関数を全て実行
        for callback in self.on_scale_changed:
            callback()
    @property
    def local_vertices(self):
        if self._local_vertices is None:
            raise RuntimeError("Transform2D: local_vertices is not set.")
        return self._local_vertices
    @local_vertices.setter
    def local_vertices(self,value:list[list[float]]):
        self._local_vertices = np.array(value, dtype=np.float32)
        self._is_dirty = True
        self._world_vertices_cache = None
    
    def _update_matrix(self):
        if not self._is_dirty:
            return
        c, s = np.cos(self.angle), np.sin(self.angle)
        x, y = self.pos.x, self.pos.y

        #モデル行列の計算,スケールは頂点座標計算時に適応
        self._matrix = np.array([
            [c, -s, x],
            [s,  c, y],
            [0,  0, 1]
        ], dtype=np.float32)

        #物体の移動を計算するためにあらかじめ、逆行列を保持しておく、この場合明示的に計算した方が効率的
        self._inv_matrix = np.array([
            [ c, s, -(x*c + y*s)],
            [-s, c,  (x*s - y*c)],
            [ 0, 0,  1]
        ], dtype=np.float32)
        self._is_dirty = False

    def _update_world_vertices(self):
        self._update_matrix()
        # 1. ローカル頂点にスケールを適用
        # local_vertices: (N, 2), scale: (1, 2) のブロードキャスト
        scaled_points = self.local_vertices * np.array([self.scale.x, self.scale.y], dtype=np.float32)
        
        # 2. 同次座標系 [x, y, 1] に変換
        n = scaled_points.shape[0]
        points_h = np.hstack([scaled_points, np.ones((n, 1), dtype=np.float32)])
        
        # 3. 剛体行列を適用してワールド座標へ
        world_points_np = (points_h @ self._matrix.T)[:, :2]

        # 4. 使いやすいように pygame.math.Vector2 のリストに変換して保存
        self._world_vertices_cache = [pygame.math.Vector2(p[0], p[1]) for p in world_points_np]

    @property
    def matrix(self):
        self._update_matrix()
        return self._matrix
    
    @property
    def inv_matrix(self):
        self._update_matrix()
        return self._inv_matrix
    @property
    def world_vertices(self):
        if self._world_vertices_cache is None:
            self._update_world_vertices()
        return self._world_vertices_cache
    
    def to_local_pos(self, world_pos: pygame.math.Vector2) -> pygame.math.Vector2:
        """ワールド座標を、スケールを無視したローカル座標（回転・移動のみ逆適用）に変換"""
        p = np.array([world_pos.x, world_pos.y, 1.0], dtype=np.float32)
        local_p = self.inv_matrix @ p
        return pygame.math.Vector2(local_p[0], local_p[1])

    def to_world_pos(self, local_pos: pygame.math.Vector2) -> pygame.math.Vector2:
        """ローカル座標（物理距離）をワールド座標に変換"""
        p = np.array([local_pos.x, local_pos.y, 1.0], dtype=np.float32)
        world_p = self.matrix @ p
        return pygame.math.Vector2(world_p[0], world_p[1])
    
    def to_world_vec(self, local_vec: pygame.math.Vector2) -> pygame.math.Vector2:
        """ローカルのベクトル（法線など）をワールドの向きに回転（スケールなし）"""
        v = np.array([local_vec.x, local_vec.y], dtype=np.float32)
        #ベクトルの場合位置は関係ないので、モデル行列をスライスして、計算
        world_vec = self.matrix[:2, :2] @ v
        return pygame.math.Vector2(world_vec[0], world_vec[1])
    
    def to_world_vectors(self, local_vectors:list[pygame.math.Vector2]) -> list[pygame.math.Vector2]:
        """ローカルのベクトル（法線など）リストをワールドの向きに回転（スケールなし）"""
        if not local_vectors:
            return []
        #入力を行列に変換
        v_array = np.array([[v.x, v.y] for v in local_vectors],dtype=np.float32)

        #まとめた行列をモデル行列の回転部分だけを使って回転
        world_v_array = v_array @ self.matrix[:2, :2].T

        #行列からリストに変換して返す。
        return [pygame.math.Vector2(row[0], row[1]) for row in world_v_array]
    
    def get_projections(self, axis: pygame.math.Vector2) -> list[float]:
        '''
        ある軸に対するワールド頂点の写像した値の頂点リストを返します。
        :param axis:写像する分離軸
        :type axis:pygame.math.Vector2
        
        :return: 対象の軸に写像された頂点のリスト
        :rtype: list[float]
        '''
        # 軸を配列化 (呼び出し側でaxis_npにして渡す設計ならさらに高速化可能)
        axis_np = [axis.x, axis.y]
        
        # 全頂点との内積 (N次元配列)
        projections_np = np.dot(self.world_vertices, axis_np)

        return projections_np.tolist()

# --- 物理コア部分 ---
class RigidBody2D:
    def __init__(self, *, mass=1.0, pos=None, restitution=0.8, friction = 0.5, lifetime=None, use_gravity=True, angle = 0.0, linear_damping=0.01, angular_damping=0.01, color=(200,80,80)):
        '''
        物理的な挙動をする剛体オブジェクトを生成するクラスです。
        
        :param mass: 物体の質量
        :param pos: 物体の位置
        :param restitution: 物体の反発係数
        :param friction: 物体の摩擦係数
        :param lifetime: 物体の存在時間
        :param use_gravity: 物体に重力を適用するかどうか
        :param angle: 物体の回転角
        :param linear_damping: 速度減衰係数
        :param angular_damping: 角速度減衰係数
        :param color: 物体の色(このパラメータは将来的に他のクラスに分離した方がいい)
        '''
        #--位置、回転、スケールの基本情報--
        self.transform = Transform2D(pos, angle) #物体の位置、回転、スケールを保持しています。

        #--質量、慣性モーメント-
        self.mass = mass #物体の質量
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0 #物体の逆質量
        self.inertia = 0.0 #物体の慣性モーメント
        self.inv_inertia = 0.0 #物体の逆慣性モーメント

        #--物理パラメータ--
        self.use_gravity = use_gravity #物体に重力を適用するかどうか
        self.restitution = restitution #反発係数
        self.friction = friction 
        self.linear_damping = linear_damping #並進速度の減衰値
        self.angular_damping = angular_damping #回転速度の減衰値

        #--運動パラメータ--
        #内部パラメータ
        self._velocity = pygame.math.Vector2(0,0) 
        self._angular_velocity = 0.0
        #速度変化オブザーバー
        self.velocity_observers = []
        self.angular_velocity_observers = []
        
        #--レンダリング--
        self.color = color

        #--ライフタイム、消滅--
        self.lifetime = lifetime
        self._is_dead = False
    
    @property
    def velocity(self):
        '''
        セッターを通過するため、Vector2のメゾット( vec.x = )や( vec.update(value) )などで書き換えないこと
        '''
        return self._velocity
    @velocity.setter
    def velocity(self,value):
        self._velocity.update(value)
        #速度変化に対して、購読している関数を実行する。
        for callback in self.velocity_observers:
            callback()
    @property
    def angular_velocity(self):
        return self._angular_velocity
    @angular_velocity.setter
    def angular_velocity(self,value):
        self._angular_velocity = value
        #角速度変化に対して、購読している関数を実行する。
        for callback in self.angular_velocity_observers:
            callback()

    def update(self, dt:float, settings:PhysicsSettings):
        '''
        オブジェクトの速度、角速度に基づいて、物体の位置と角度を更新します。デルタタイム間は等速運動として、運動を近似します。
        
        :param dt: デルタタイム
        :type dt: float
        :param settings: 物理シミュレーションの設定(速度閾値を利用)
        :type settings: PhysicsSettings
        '''
        if self.lifetime is not None:
            self.lifetime -= dt
            if self.lifetime <= 0:
                self._is_dead = True

        if self.inv_mass > 0:
            # 速度をわずかに減衰させる
            self.velocity *= (1.0 - self.linear_damping * dt)
            # 閾値判定（二乗比較）閾値以下なら微小な速度変化を０に丸め込む
            if self.velocity.length_squared() < settings.velocity_threshold ** 2:
                self.velocity.update(0, 0)
            if self.velocity.x != 0.0 or self.velocity.y != 0.0:
                # 速度がゼロでないなら、位置を更新
                self.transform.pos += self.velocity * dt

        if self.inv_inertia > 0:
            #角速度をわずかに減衰させる
            self.angular_velocity *= (1.0 - self.angular_damping * dt)
            #角速度が閾値以下なら０に丸め込み
            if abs(self.angular_velocity) < settings.angular_threshold:
                self.angular_velocity = 0.0

            if self.angular_velocity != 0.0:
                #角速度を元に、角度を更新
                self.transform.angle += self.angular_velocity * dt

    def apply_force(self,force:pygame.math.Vector2,dt: float):
        """継続的な力を加える（重力など）"""
        self.apply_impulse(force * dt)
    
    def apply_impulse(self, impulse:pygame.math.Vector2):
        """瞬間的な衝撃を加える（衝突など）"""
        # 速度を直接変化させる： v = v + J / m
        if self.inv_mass > 0:
            self.velocity += impulse * self.inv_mass

    def apply_impulse_at_offset(self, impulse: pygame.math.Vector2, r: pygame.math.Vector2):
        '''
        重心以外の特定の地点に撃力を加えます。接触時間は微小なものと考え、直接速度に加算します。（並進 + 回転）。
        
        :param impulse: 加える撃力ベクトル (J)
        :type impulse: pygame.math.Vector2
        :param r: 重心から衝突点へのワールド空間での相対ベクトル (r - pos)
        :type r: pygame.math.Vector2
        '''
        if self.inv_mass == 0:
            return pygame.math.Vector2(self.velocity)

        # 並進運動として重心を動かす
        self.apply_impulse(impulse)

        # 指定された位置にトルクを加える
        if self.inv_inertia > 0:
            # 2D外積: r × J
            angular_impulse = r.x * impulse.y - r.y * impulse.x
            self.angular_velocity += angular_impulse * self.inv_inertia

    def apply_constraint_impulse(self, linear_jacobian: pygame.math.Vector2, angular_jacobian: float, lambda_val: float):
        '''
        拘束ソルバーから受け取ったヤコビアン成分と力積(λ)の大きさに基づいて、速度と角速度を更新します。
        
        :param linear_jacobian: 並進方向のヤコビアン成分 (J_v)
        :param angular_jacobian: 回転方向のヤコビアン成分 (J_w)
        :param lambda_val: 適用する力積の大きさ (λ または Δλ)
        '''
        if self.inv_mass > 0:
            self.velocity += linear_jacobian * (lambda_val * self.inv_mass)
        if self.inv_inertia > 0:
            self.angular_velocity += angular_jacobian * (lambda_val * self.inv_inertia)
    
    def get_jacobian_velocity(self, linear_jacobian: pygame.math.Vector2, angular_jacobian: float) -> float:
        '''
        与えられたヤコビアン（並進・回転成分）に対する、この剛体の現在の速度の投影成分（JV）を計算して返します。
        
        :param linear_jacobian: 並進方向のヤコビアン成分 (J_v)
        :param angular_jacobian: 回転方向のヤコビアン成分 (J_w)
        :return: ヤコビアン方向の速度成分 (float)
        '''
        return linear_jacobian.dot(self.velocity) + angular_jacobian * self.angular_velocity
    
    def get_velocity_at_offset(self, r:pygame.math.Vector2) -> pygame.math.Vector2:
        '''
        重心からのオフセットベクトル r 地点における速度（並進 + 回転）を返します。
        
        :param r: 重心から対象点へのワールド空間での相対ベクトル
        :type r: pygame.math.Vector2
        :return: その地点における速度ベクトル
        :rtype: pygame.math.Vector2
        '''
        if self.inv_inertia == 0:
            return pygame.math.Vector2(self.velocity)
        
        # 回転によって発生する接線速度ベクトルを求める
        # 位置ベクトルrと角速度の外積を計算すればいい
        tangential_velocity = pygame.math.Vector2(-self.angular_velocity * r.y, self.angular_velocity * r.x)
        return self.velocity + tangential_velocity

class Region:
    """
    軸並行な矩形領域のクラスです。４分木などのブロードフェーズの際に参照されます。
    """
    def __init__(self, pos:pygame.math.Vector2, half_w:float, half_h:float):
        '''
        軸並行領域を作成

        :param pos: 領域の位置ベクトル
        :param half_w: 領域の横幅の半分
        :param half_h: 領域の高さの半分
        '''
        self.pos = pos
        self.half_width = half_w
        self.half_height = half_h

    @property
    def left(self): return self.pos.x - self.half_width
    @property
    def right(self): return self.pos.x + self.half_width
    @property
    def top(self): return self.pos.y - self.half_height
    @property
    def bottom(self): return self.pos.y + self.half_height

    def intersects(self, other:"Region") -> bool:
        '''
        軸並行領域が重なっているかどうかを判定し、真偽値を返します。
        '''
        return not (other.left > self.right or
                    other.right < self.left or
                    other.top > self.bottom or
                    other.bottom < self.top)

class ShapeType(Enum):
    
    CIRCLE = auto()
    RECT = auto()

class CollisionObject(RigidBody2D):
    """
    形を持つ剛体のための抽象クラス
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #ブロードフェーズに用いる軸並行な領域のキャッシュ
        self._region_cache:Region = Region(self.transform.pos,0.0,0.0)
        #リージョンのキャッシュの更新が必要かどうかのフラグ
        self._is_region_dirty = True
        #４分木における登録された空間のアドレス
        self.addresses = []
        #スケール変更を検知するためにオブザーバに登録。スケール変化時に境界距離やモーメントに更新が必要なため
        self.transform.on_scale_changed.append(self._update_inertia)
        self.transform.on_scale_changed.append(self._mark_region_dirty)
        self._update_inertia()

    @property
    def shape_type(self) -> ShapeType:
        """子クラスで必ず実装し、固有の形状を返す"""
        raise NotImplementedError
    
    @property
    def local_axes(self) -> list[pygame.math.Vector2]:
        '''オブジェクトの各辺に対する、互いに並行でない法線ベクトルを返します。主に分離軸の計算に使用します。'''
        return []
    
    @property
    def region(self) -> Region:
        '''オブジェクトを囲む軸並行な領域を返す'''
        if self._is_region_dirty:
            self._update_region()
            self._is_region_dirty = False
        return self._region_cache
    
    def get_neighbor_vertices_index(self, index: int) -> tuple[int, int]:
        """
        指定した頂点インデックスに対して、前(index-1)と次(index+1)の
        頂点インデックスを巡回（ループ）を考慮して返します。
        """
        # 矩形なら 4 ですが、汎用的に頂点数で計算します
        count = len(self.transform.world_vertices)
        
        prev_idx = (index - 1) % count
        next_idx = (index + 1) % count
        
        return prev_idx, next_idx

    def get_edge_vertices_index(self, index: int) -> list[int]:
        """
        指定したインデックス i の頂点と、その次の頂点 (i+1)%N のインデックスを返します。
        """
        count = len(self.transform.world_vertices)
        return [index, (index + 1) % count]
    
    def _update_region(self):
        '''軸並行領域を更新する'''
        raise NotImplementedError
    
    def _mark_region_dirty(self):
        '''リージョンの更新フラグを有効にします。'''
        self._is_region_dirty = True

    def _update_inertia(self):
        """形状と質量に基づき慣性モーメントを計算・設定する"""
        raise NotImplementedError

class CircleBody(CollisionObject):
    def __init__(self, radius=20, segments=32, **kwargs):
        super().__init__(**kwargs)
        #単位円の時の頂点行列を定義
        self.transform.local_vertices = self._generate_unit_circle(segments)
        #x,yスケール値を半径に設定
        self.transform.scale = (radius,radius)

    @property
    def shape_type(self) -> ShapeType:
        return ShapeType.CIRCLE
    
    def _generate_unit_circle(self, n):
        #0~2piをn分割して、x,y座標を計算する。
        theta = np.linspace(0,2*np.pi, n, endpoint=False)
        return np.stack([np.cos(theta), np.sin(theta)], axis=1)
    
    def _update_region(self):
        radius = self.transform.scale.x
        self._region_cache.half_width = radius
        self._region_cache.half_height = radius
    
    def _update_inertia(self):
        if self.mass > 0:
            # 円盤の慣性モーメント: I = 0.5 * mass * r^2
            self.inertia = 0.5 * self.mass * (self.transform.scale.x ** 2)
            self.inv_inertia = 1.0 / self.inertia
        else:
            self.inertia = 0.0
            self.inv_inertia = 0.0

class RectBody(CollisionObject):
    # ---------------------------------------------------------
    # 頂点定義 (CCW: 反時計回り)
    # Pygame座標系 (Y-Down) を前提とします。
    # ---------------------------------------------------------
    _RECT_VERTICES = np.array([
        [-0.5, -0.5], # 0: TopLeft     (左上)
        [ 0.5, -0.5], # 1: TopRight    (右上)
        [ 0.5,  0.5], # 2: BottomRight (右下)
        [-0.5,  0.5]  # 3: BottomLeft  (左下)
    ], dtype=np.float32)

    # ---------------------------------------------------------
    # 軸 (法線) の定義
    # Axis[i] は Vertex[i] から Vertex[i+1] への辺の「外向き法線」
    # ---------------------------------------------------------
    _LOCAL_AXES = [
        # Index 0 (Vertex 0->1): 上面 (Top Face)
        # Vector(1, 0) x (0, 0, 1) -> (0, -1) ※Y下向き座標系での「上」
        pygame.math.Vector2(0, -1), 
        
        # Index 1 (Vertex 1->2): 右面 (Right Face)
        pygame.math.Vector2(1, 0),
        
        # Index 2 (Vertex 2->3): 下面 (Bottom Face)
        pygame.math.Vector2(0, 1),
        
        # Index 3 (Vertex 3->0): 左面 (Left Face)
        pygame.math.Vector2(-1, 0)
    ]

    def __init__(self, scale_x=100, scale_y=20, **kwargs):
        super().__init__(**kwargs)
        self.transform.local_vertices = RectBody._RECT_VERTICES
        self.transform.scale = (scale_x,scale_y)
        #多角形の場合、角度変化時にもリージョンを更新する必要があるので、角度変化を購読
        self.transform.on_angle_changed.append(self._mark_region_dirty)
        
    @property
    def shape_type(self) -> ShapeType:
        return ShapeType.RECT
    
    @property
    def local_axes(self):
        return self._LOCAL_AXES
    
    @staticmethod
    def get_rect_feature(local_pos:pygame.math.Vector2, hw:float, hh:float) -> tuple:
        """
        ローカル座標点が矩形のどのフィーチャー(頂点V/辺E)に位置するかを判定する。
        ローカル座標点はあらかじめ [-hw, hw], [-hh, hh] の範囲にクランプされている前提。
        """
        # 境界に接しているかどうかのフラグ (-1: 負の端, 1: 正の端, 0: 内部)
        # EPSILON を使うと浮動小数点誤差に強くなる
        eps = 1e-6
        sign_x = -1 if local_pos.x <= -hw + eps else 1 if local_pos.x >= hw - eps else 0
        sign_y = -1 if local_pos.y <= -hh + eps else 1 if local_pos.y >= hh - eps else 0

        # 1. 頂点の判定 (両方の軸が端にある)
        if sign_x != 0 and sign_y != 0:
            if sign_x == -1 and sign_y == -1: return ('V', 0) # 左上
            if sign_x ==  1 and sign_y == -1: return ('V', 1) # 右上
            if sign_x ==  1 and sign_y ==  1: return ('V', 2) # 右下
            if sign_x == -1 and sign_y ==  1: return ('V', 3) # 左下

        # 2. 辺の判定 (片方の軸だけが端にある)
        if sign_y == -1: return ('E', 0) # 上辺
        if sign_x ==  1: return ('E', 1) # 右辺
        if sign_y ==  1: return ('E', 2) # 下辺
        if sign_x == -1: return ('E', 3) # 左辺

        # 3. ど真ん中（内部）の場合：暫定で上辺とするか、エラーを返す
        return ('E', 0)
    
    def _update_region(self):
        # 既に計算済みのワールド頂点リストを取得
        vertices = self.transform.world_vertices
        
        # 直接 x と y のリストを抽出（内包表記で爆速）
        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]

        # 差分を計算してキャッシュを更新
        self._region_cache.half_width = (max(xs) - min(xs)) / 2.0
        self._region_cache.half_height = (max(ys) - min(ys)) / 2.0
    
    def _update_inertia(self):
        if self.mass > 0:
             # 長方形の慣性モーメント: I = (1/12) * mass * (width^2 + height^2)
            w, h = self.transform.scale.x, self.transform.scale.y
            self.inertia = (1.0 / 12.0) * self.mass * (w**2 + h**2)
            self.inv_inertia = 1.0 / self.inertia
        else:
            self.inertia = 0.0
            self.inv_inertia = 0.0

class MortonManager:
    '''
    モートン番号を使った、線形四分木を管理するクラス
    '''
    #深さの上限、最大16まで対応
    DEPTH_LIMIT = 16
    def __init__(self,region:Region,max_depth=8):
        '''
        モートンマネージャーを生成

        :param region: 分割する空間領域
        :type region: Region
        :param max_depth: 分割する最大深さ(このクラスは0~16までを想定)
        '''
        self.width, self.height = region.half_width * 2, region.half_height * 2
        #分割計算時のため、短い方の幅を計算しておく
        self.short_side = min(self.width, self.height)
        self.world_region = region
        self.max_depth = max(0,min(max_depth, self.DEPTH_LIMIT))
        #軸の分割数2^max_depth
        self.grid_count = 1 << max_depth
        self.spatial_map: dict[tuple,list] = {}

    def get_addresses(self,region:Region):
        '''
        物体の軸並行領域が所属する空間分割上のアドレスを取得し、設定する。

        :param region: オブジェクトの軸並行領域
        :type region: Region
        '''
        #領域が分割空間と重なっていないなら、即抜ける。
        if not self.world_region.intersects(region):
            return  []
        
        #領域の大きい方の幅を取得する。
        size = max(region.half_width * 2, region.half_height * 2)

        #サイズが０より小さいなら、最も深い(細かい)階層を割り当てる
        if size <= 0:
            level = self.max_depth
        #サイズに応じて適切な階層を計算する。
        else:
            level = int(math.log2(self.short_side/size))
        #計算した階層を、0~maxの範囲でクランプする。
        level = max(0,min(level, self.max_depth))
        #オブジェクトが登録される階層における、分割数を計算する。
        grid_count = 1 << level
        
        #リージョンの左上と右下の頂点をグリット座標へ変換
        gx1 = int(max(0, min(region.left, self.width -1)) / self.width * grid_count)
        gy1 = int(max(0, min(region.top, self.height -1)) / self.height * grid_count)
        gx2 = int(max(0, min(region.right, self.width -1)) / self.width * grid_count)
        gy2 = int(max(0, min(region.bottom, self.height -1)) / self.height * grid_count)

        #オブジェクトが登録される空間アドレスを取得
        addresses = []
        for y in range(gy1, gy2+1):
            for x in range(gx1, gx2+1):
                morton_id = self._interleave(x,y)
                addresses.append((level,morton_id))
        return addresses

    def clear(self):
        '''オブジェクトとアドレスの辞書を削除する'''
        self.spatial_map = {}

    def register(self, entity: CollisionObject):
        '''オブジェクトの分割空間上のアドレスを取得し、アドレスをキーとした辞書に登録する。'''
        addrs = self.get_addresses(entity.region)
        entity.addresses = addrs

        for addr in addrs:
            if addr not in self.spatial_map:
                self.spatial_map[addr] = []
            self.spatial_map[addr].append(entity)
        
        return addrs
    
    def get_collision_candidates(self, entity:CollisionObject) -> list[CollisionObject]:
        '''オブジェクトと衝突する可能性がある他のオブジェクトを取得する。'''
        if not entity.addresses:
            return []
        candidates = set()
        for curr_level, curr_id in entity.addresses:
            #親方向の探索
            temp_level, temp_id = curr_level, curr_id
            while temp_level >= 0:
                key = (temp_level, temp_id)
                if key in self.spatial_map:
                    candidates.update(self.spatial_map[key])
                #階層を一つ上げる
                temp_level -= 1
                #モートン番号を一つ上の階層の番号に変換
                temp_id >>= 2
        #自分自身をセットから安全に削除
        candidates.discard(entity)

        return list(candidates)
    
    def draw_grid(self, surface):
        """空間分割のグリッドを描画する"""
        for depth in range(1, self.max_depth + 1):
            # その階層での分割数 (Depth 1なら2分割、Depth 2なら4分割...)
            divisions = 1 << depth
            cell_w = self.width / divisions
            cell_h = self.height / divisions
            
            # 階層ごとに線の色と太さを変える
            if depth == 1:
                color = (150, 150, 150)  # Level 1の境界（画面のド真ん中十字）
                thickness = 3
            elif depth == 2:
                color = (80, 80, 80)     # Level 2の境界
                thickness = 2
            else:
                color = (40, 40, 40)     # Level 3以降の細かい境界
                thickness = 1

            for i in range(1, divisions):
                # 偶数番目の線は、より浅い階層（太い線）で既に描画されているのでスキップ
                if i % 2 == 0:
                    continue
                
                # 縦線
                x = i * cell_w
                pygame.draw.line(surface, color, (x, 0), (x, self.height), thickness)
                # 横線
                y = i * cell_h
                pygame.draw.line(surface, color, (0, y), (self.width, y), thickness)

    def _interleave(self,x:int,y:int):
        '''グリッド上のx,y座標からモートン番号を作成する。'''
        #yを一つシフトして、yxyxyx...のような並びを作る
        return (self._part1by1(y) << 1) | self._part1by1(x)

    def _part1by1(self, n):
        # どんな深さ(最大16)でも、これで「1つ飛ばしの隙間」が作れる
        n &= 0x0000ffff                 # 16ビットより上を掃除
        n = (n | (n << 8)) & 0x00ff00ff # [8ビットの塊] 0000000011111111...
        n = (n | (n << 4)) & 0x0f0f0f0f # [4ビットの塊] 0000111100001111...
        n = (n | (n << 2)) & 0x33333333 # [2ビットの塊] 0011001100110011...
        n = (n | (n << 1)) & 0x55555555 # [1ビットの塊] 0101010101010101...
        return n

@dataclass
class ContactPoint:
    '''
    接触点とそののめり込みの深さをセットで保存するデータクラス

    Attributes:
        _id (tuple): 
            この接触点の「出自」を表す一意識別子（Feature ID）。
            フレームを跨いで同じ接触かを判定し、力積の累積値（Warm Start）を引き継ぐために使用する。
            
            【IDの構成規則: 4要素のタプル】
            (id(body1), type_1, index_1,id(body2), type_2, index_2)
            
            - type_x : 形状の要素タイプ ('V': 頂点 / 'E': 辺)
            - index_x: 各物体内における要素のインデックス
            
            【正規化ルール (Canonical Order)】
            InteractionLink内で管理するため、常に以下の順序で記録すること：
            - 前半3つ: ID（メモリアドレス等）が小さい方の物体(Body1)の要素
            - 後半3つ: IDが大きい方の物体(Body2)の要素
            
            【具体的なIDの例】
               Body1の頂点3 と Body2の辺5 が接触している場合:
               id = (id(body1),'V', 3,id(body2), 'E', 5)
        
        point (pygame.math.Vector2): 接触地点のワールド座標。
        depth (float): その点における法線方向のめり込み深さ（正の値）。
    '''
    point: pygame.math.Vector2 # ワールド座標
    depth: float = 0.0         # その点におけるめり込み深さ
    # init=False にすることで、生成時の引数から除外されます
    _id: tuple = field(init=False, default=None) # 頂点のID(どの物体間のどの点か)　キャッシュ検索用

    @property
    def id(self):
        return self._id
    
    def generate_feature_id(self, body_1:CollisionObject, body_2:CollisionObject, feature_1:tuple, feature_2:tuple):
        '''
        その点におけるidを作成します。body1,2の参照値を基準に、順序を決定するので、入力順によらず二つの物体間の接触点のidは正規化されます。
        
        :param body_1: 接触点を構成するオブジェクト１
        :type body_1: CollisionObject
        :param body_2: 接触点を構成するオブジェクト２
        :type body_2: CollisionObject
        :param feature_1: (type,index)のタプル 例('V', 2)
        :type feature_1: tuple
        :param feature_2: (type,index)のタプル 例('E', 0)
        :type feature_2: tuple
        '''
        id_1 = id(body_1)
        id_2 = id(body_2)
       # ボディの固有ID（整数値）の大小を基準に、常に (小, 大, 小の特徴, 大の特徴) の順に並べる
        if id_1 < id_2:
            self._id = (id_1, id_2, feature_1, feature_2)
        else:
            self._id = (id_2, id_1, feature_2, feature_1)

@dataclass
class CollisionResult:
    '''
    コリジョン判定の結果を保存するためのデータクラスです。
    衝突したかどうか、衝突面の法線ベクトル、のめり込みの深さを格納します。
    '''
    collided: bool = False
    normal: pygame.math.Vector2 = None # origin から target への向き
    contact_points: list[ContactPoint] = field(default_factory=list)
    #法線ベクトルの始点
    body_origin: CollisionObject = None
    #法線ベクトルの終点
    body_target: CollisionObject = None

@dataclass
class SATResult:
    """SAT処理に用いる中間データ"""
    is_separated: bool = False                  #二つのオブジェクト間に分離線が引けるかどうか
    min_overlap: float = float('inf')     # 最小の重なり
    best_axis: pygame.math.Vector2 = None # ワールド空間での法線
    ref_body: CollisionObject = None      # この判定における基準体
    inc_body: CollisionObject = None      # この判定における入射体
    ref_axis_index: int = -1              # 基準体のどの辺か
    inc_deep_index: int = -1              # 入射体のどの頂点が一番深いか
    ref_offset: float = 0.0               # 入射軸の法線ベクトルに対する内積情報を保持
    inc_projections: np.ndarray = None    # 入射体の分離軸における、入射体の頂点写像のキャッシュ
    ref_face_index: list[int] = None      # 基準面を構成する２頂点のインデックス[v1,v2]
    inc_face_index: list[int] = None      # 入射面を構成する２頂点のインデックス[v1,v2]

class CollisionManager:
    # --クラス定数--
    #丸め込み閾値を定義
    EPSILON = 1e-3
    #丸め込み閾値の二乗
    EPSILON_SQ = EPSILON ** 2
    def __init__(self):
        #形状ごとの判定関数の
        self._strategies = {}

        #デフォルトの形状対別処理設定
        self.register_strategy(ShapeType.CIRCLE, ShapeType.CIRCLE, self.solve_circle_circle)
        self.register_strategy(ShapeType.CIRCLE, ShapeType.RECT, self.solve_circle_rect)
        self.register_strategy(ShapeType.RECT, ShapeType.RECT, self.solve_SAT_rect_rect)
    
    def register_strategy(self, shape_a:ShapeType, shape_b:ShapeType, func):
        '''
        形状のペアごとに、衝突アルゴリズムを設定します。
        '''
        self._strategies[(shape_a, shape_b)] = func 
    
    def solve(self, b1:CollisionObject, b2:CollisionObject) -> CollisionResult:
        #お互いの軸並行領域が重なっているかどうかを判定する。
        if not b1.region.intersects(b2.region):
            return CollisionResult(False)
        
        #衝突している可能性があるオブジェクト同士に衝突判定を適応する。
        key = (b1.shape_type, b2.shape_type)
        if key in self._strategies:
            res = self._strategies[key](b1, b2)
            return res
        
        reverse_key = (b2.shape_type, b1.shape_type)
        if reverse_key in self._strategies:
            res:CollisionResult = self._strategies[reverse_key](b2, b1)
            return res
        
        #該当するアルゴリズムが設定されていない時、Falseを返す。
        return CollisionResult(False)
    
    def solve_circle_circle(self, c1:CircleBody, c2:CircleBody) -> CollisionResult:
        '''
        二つの円形オブジェクトの接触を判定します。
        
        :param c_1: 対象となる円形オブジェクト ターゲット側
        :type c_1: CircleBody
        :param c_2: 対象となる円形オブジェクト オリジン側
        :type c_2: CircleBody
        :return: 接触しているかどうか、接触面の法線ベクトル、のめり込みの深さ、接触点の位置を返します。
        :rtype: CollisionResult
        '''
        result = CollisionResult(collided=False)

        #二つの円の位置座標の差から距離を計算する。
        diff:pygame.math.Vector2 =  c1.transform.pos - c2.transform.pos
        #diffは衝突の法線ベクトルなので、オリジンとターゲットが確定する。
        result.body_origin, result.body_target = c2, c1
        #衝突判定では二乗の距離で判定
        dist_sq = diff.length_squared()
        radius_sum = c1.transform.scale.x + c2.transform.scale.x
        #二つの円距離が二つの円の半径の和より小さいなら接触。ゼロ除算を避けるために距離が０である場合は避ける。
        if dist_sq < radius_sum ** 2:
            result.collided = True
            distance = diff.length()
            if distance == 0:
                #向きを特定できないので、上方向で丸め込む
                result.normal = pygame.math.Vector2(0,-1)
                overlap = radius_sum
            else:
                result.normal = diff.normalize()
                overlap = radius_sum - distance
            #二つの円の法線方向の中間点を接触点とする
            point = c2.transform.pos + result.normal * (c2.transform.scale.x + overlap * 0.5)
            #コンタクトポイントを作成、円同士の衝突は一点
            contact = ContactPoint(point,overlap)
            #コンタクトポイントのidを作成。円の場合構成要素は必ず(E,0)とする。
            contact.generate_feature_id(c1,c2,('E',0),('E',0))
            #衝突点のリストに追加
            result.contact_points.append(contact)
        return result
    
    def solve_circle_rect(self, circle:CircleBody, rect:RectBody) -> CollisionResult:
        '''
        円形オブジェクトと長方形オブジェクトとの接触を判定します。
        
        :param circle: 対象となる円形オブジェクト ターゲット側
        :type circle: CircleBody
        :param rect: 対象となる長方形オブジェクト オリジン側
        :type rect: RectBody
        :return: 接触しているかどうか、接触面の法線ベクトル、のめり込みの深さ、接触点を返します。
        :rtype: CollisionResult
        '''
        #返り値を用意
        result = CollisionResult(collided=False)

        #円を矩形側を基準とするローカル座標に変換する。
        circle_local_pos:pygame.math.Vector2 = rect.transform.to_local_pos(circle.transform.pos)

        #矩形の幅と高さを取得
        hw,hh = rect.transform.scale.x/2, rect.transform.scale.y/2

        # 最短距離の点を求める（クランプ処理）
        closest_x = max( -hw ,min(circle_local_pos.x, hw))
        closest_y = max( -hh, min(circle_local_pos.y, hh))
        closest_vec = pygame.math.Vector2(closest_x, closest_y)
        
        diff:pygame.math.Vector2 = circle_local_pos - closest_vec
        #diffの方向が衝突の法線ベクトルになりうるので、オリジンとターゲットの関係が定まる。
        result.body_origin, result.body_target = rect, circle
        distance = diff.length()
          
        if 0 < distance <= circle.transform.scale.x:
            result.collided = True
            local_normal = diff.normalize()
            result.normal = rect.transform.to_world_vec(local_normal).normalize()
            overlap = circle.transform.scale.x - distance
            # 接触点はクランプした点をワールドに戻したもの
            point = rect.transform.to_world_pos(closest_vec)
            #矩形のフィーチャー情報を取得
            rect_feature =rect.get_rect_feature(closest_vec, hw, hh)
            #コンタクトポイントを作成、この場合も一つだけ
            contact = ContactPoint(point,overlap)
            #IDの作成、円側は常に('E',0)に固定する。
            contact.generate_feature_id(rect,circle,rect_feature,('E',0))
            #衝突点を登録
            result.contact_points.append(contact)

        elif distance <= self.EPSILON: # 中心が中、あるいは重なっている
            result.collided = True
            # 左右の壁、上下の壁、どこが一番近いか計算
            dist_left = circle_local_pos.x + hw
            dist_right = hw - circle_local_pos.x
            dist_top = circle_local_pos.y + hh
            dist_bottom = hh - circle_local_pos.y

            # 一番近い方向を選んで、そこを法線(normal)とする
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            if min_dist == dist_top:
                local_normal = pygame.math.Vector2(0, -1)
                rect_feature = ('E', 0) # 上
            elif min_dist == dist_right:
                local_normal = pygame.math.Vector2(1, 0)
                rect_feature = ('E', 1) # 右
            elif min_dist == dist_bottom:
                local_normal = pygame.math.Vector2(0, 1)
                rect_feature = ('E', 2) # 下
            else: # dist_left
                local_normal = pygame.math.Vector2(-1, 0)
                rect_feature = ('E', 3) # 左
            result.normal = rect.transform.to_world_vec(local_normal).normalize()
            overlap = circle.transform.scale.x + min_dist
            # 内部にいる場合、接触点は円の中心から法線方向に半径分戻ったあたり
            point = circle.transform.pos - result.normal * circle.transform.scale.x
             #コンタクトポイントを作成、この場合も一つだけ
            contact = ContactPoint(point,overlap)
            #IDの作成、円側は常に('E',0)に固定する。
            contact.generate_feature_id(rect,circle,rect_feature,('E',0))
            #衝突点を登録
            result.contact_points.append(contact)
        return result
    
    def solve_SAT_rect_rect(self, r1:CollisionObject, r2:CollisionObject) -> CollisionResult:
        '''
        二つの矩形オブジェクトの接触を分離軸定理(SAT)を用いて判定します。二つの辺の数の和をNとすると、
        最大でO(N)の計算量が発生します。
        
        :param rect_1: 対象となる矩形オブジェクト１
        :type rect_1: RectBody
        :param rect_2: 対象となる矩形オブジェクト２
        :type rect_2: RectBody
        :return: 接触しているかどうか、接触面の法線ベクトル、のめり込みの深さを返します。
        :rtype: CollisionObject
        '''
        #戻り値用の最終結果を用意
        collision_result = CollisionResult(True)

        #オブジェクト１を基準に判定
        result_1 = self._run_one_way_SAT(r1, r2)
        if result_1.is_separated:
            return CollisionResult(False)
        
        #オブジェクト２を基準に判定
        result_2 = self._run_one_way_SAT(r2, r1)
        if result_2.is_separated:
            return CollisionResult(False)
        
        # 重なりが「より小さい（浅い）」方を最終的な衝突情報として採用
        sat_result = result_1 if result_1.min_overlap < result_2.min_overlap else result_2
        
        #基準面を取得
        sat_result.ref_face_index = sat_result.ref_body.get_edge_vertices_index(sat_result.ref_axis_index)

        #最も深い点を含む二つの辺を比較して、より内積が小さい方を入射面とする。
        sat_result.inc_face_index = self._get_incident_face(sat_result)

        #入射面と基準面の情報を使って接触点を生成
        contact_points = self._process_manifold(sat_result)

        # --SATリザルトの情報使って、コリジョンリザルトに必要な情報を埋める--
        collision_result.body_origin = sat_result.ref_body
        collision_result.body_target = sat_result.inc_body
        collision_result.normal = sat_result.best_axis
        collision_result.contact_points = contact_points
        return collision_result
         
    def _run_one_way_SAT(self, ref: CollisionObject, inc: CollisionObject) -> SATResult:
        '''
        基準オブジェクト(ref)の分離軸を用いて、入射オブジェクト(inc)に対するSATを実行します
        
        :param ref: 基準オブジェクト
        :type ref: CollisionObject
        :param inc: 入射オブジェクト
        :type inc: CollisionObject
        :return: 判定結果
        :rtype: SATResult
        '''
        #結果を収納するデータオブジェクトを設定
        result = SATResult(ref_body=ref,inc_body=inc)
        
        #基準オブジェクトのローカルベクトルをワールドベクトルに変換
        ref_axes = ref.transform.to_world_vectors(ref.local_axes)
        #基準オブジェクトのワールド頂点座標を保存
        ref_verts = ref.transform.world_vertices

        for i, axis in enumerate(ref_axes):
            # 基準側(ref)の射影 [O(1)最適化]
            # 軸iに対応する頂点iが、この軸方向での最大値（境界）
            proj_ref_max = axis.dot(ref_verts[i])

            # 入射側(inc)の分離軸に対する写像を計算
            proj_inc = inc.transform.get_projections(axis)

            #写像した頂点の最小値と、そのインデックスを取得
            proj_inc_min = min(proj_inc)
            proj_inc_min_idx = proj_inc.index(proj_inc_min)

            #オーバーラップを計算して、重なりを計算する。
            overlap = proj_ref_max - proj_inc_min

            #重なりが0より小さいなら、分離線が引けるので接触していないことが確定する。
            if overlap < 0:
                result.is_separated = True
                return result
            
            #最小の重なりを持つ分離軸を衝突面の法線ベクトルとなるので。最小値を更新していく。それに合わせて結果のパラメータも更新
            if overlap < result.min_overlap:
                result.min_overlap = overlap
                result.best_axis = axis
                result.ref_axis_index = i
                result.inc_deep_index = proj_inc_min_idx
                result.inc_projections = proj_inc
                result.ref_offset = proj_ref_max
        return result
    
    def _get_incident_face(self, result: SATResult) -> list[int]:
        '''
        SATの判定結果から得られた、入射オブジェクトと基準オブジェクトの法線ベクトルに対して、最も深い点から、最も深い点を含む二つの辺の内積を比較して、
        より小さい方を入射面として返す。
        
        :param result: SATアルゴリズムによって得られた結果(入射オブジェクト、最も深い点、基準面の法線ベクトル)
        :type result: SATResult
        :return: 入射面を構成する頂点のインデックスリスト
        :rtype: list[Vector2]
        '''
        #結果から必要なデータを取得
        inc_body = result.inc_body
        deep_idx = result.inc_deep_index
        proj_inc = result.inc_projections

        #最も深い点隣接頂点のインデックスを取得
        prev_idx, next_idx = inc_body.get_neighbor_vertices_index(deep_idx)

        #キャッシュ情報から、もっとも深い点に隣接する二点の内積値を取得する
        proj_prev = proj_inc[prev_idx]
        proj_next = proj_inc[next_idx]

        #二つの内積を比較して、より小さい方のインデックスが入射面の辺である。
        if proj_prev < proj_next:
            return inc_body.get_edge_vertices_index(prev_idx)
        else:
            return inc_body.get_edge_vertices_index(deep_idx)
        
    def _process_manifold(self, result: SATResult) -> list[ContactPoint]:
        '''
        SATの結果から、入射面と基準面の情報を受け取り、二つのオブジェクトが接触する点を返します。
        
        :param result: SATデータ(基準面、入射面などの情報)
        :type result: SATResult
        :return: 接触点の情報リスト
        :rtype: list[ContactPoint]
        '''
        # --リザルトから必要なデータを取得する。--
        ref_wv = result.ref_body.transform.world_vertices
        inc_wv = result.inc_body.transform.world_vertices
        ref_face_index = result.ref_face_index #基準面の頂点リストのインデックス[v1, v2]
        inc_face_index = result.inc_face_index #入射面の頂点リストのインデックス[v1, v2]
        ref_face = [ref_wv[i] for i in ref_face_index]
        height = result.ref_offset  #基準面の高さ
        normal = result.best_axis  #衝突面の法線ベクトル

        #入射面の辺インデックスは、最初の頂点のインデックスに等しい
        inc_edge_idx = inc_face_index[0]
        #基準面側の辺インデックスを取得。暗黙的に辺とベクトルとインデックスが等しい
        ref_edge_idx = result.ref_axis_index

        # --- 　 初期状態：コンタクトポイントを作成し、辺のリストを作る ---
        clip_face = [
            ContactPoint(inc_wv[inc_face_index[0]]),
            ContactPoint(inc_wv[inc_face_index[1]])
        ]

        #初期状態では、入射側の頂点を暫定的な接触点として、IDを作る。IDは基準側の辺と入射側の各頂点とする。
        clip_face[0].generate_feature_id(result.ref_body, result.inc_body, ('E',ref_edge_idx), ('V', inc_face_index[0]))
        clip_face[1].generate_feature_id(result.ref_body, result.inc_body, ('E',ref_edge_idx), ('V', inc_face_index[1]))

        #基準面に平行な基準ベクトルを定義する。v1基準
        s = (ref_face[1] - ref_face[0]).normalize()

        # --クリップ処理
        #基準側のv1法線sを基準にクリップ
        clip_face = self._clip_to_line(
            clip_face, s, ref_face[0],
            inc_edge_idx, ref_face_index[0],
            result.inc_body, result.ref_body
        )

        #基準側のv2でクリップ(ベクトルsはv1基準なので反転させる)
        clip_face = self._clip_to_line(
            clip_face, -s, ref_face[1],
            inc_edge_idx, ref_face_index[1],
            result.inc_body, result.ref_body
        )
        # --- めり込み深さの計算と有効な点の抽出、基準面より上の接触点を削除する（インプレース処理） ---
        # 削除によるインデックスのズレを防ぐため、後ろから順に確認する
        for i in range(len(clip_face) -1, -1, -1):
            cp = clip_face[i]
            #衝突法線に対する、内積を計算して、その点ののめり込みの深さを調べる。
            projection = normal.dot(cp.point)

            #基準面より深い点のみを有効にする
            if projection <= height:
                #のめり込みの深さは基準面の高さから接触点座標の法線への内積値の差
                cp.depth = height - projection
            #基準面より上にある場合、無効な接触点として、リストに含めない
            else:
                clip_face.pop(i)

        return clip_face

    def _clip_to_line(self, inc_face:list[ContactPoint],
                      normal:pygame.math.Vector2,
                      offset_point:pygame.math.Vector2, 
                      inc_edge_idx: int,
                      offset_idx:int,
                      inc_body:CollisionObject,
                      ref_body:CollisionObject) -> list[ContactPoint]:
        '''
        指定された単位ベクトルnormal上の境界線(offset_point)に対して、入射面の各点の内積を取り、境界を下回った点をクリップし、入射面と境界線との交点に置き換えます。
        
        :param inc_face: クリップしたい入射面
        :type inc_face: list[ContactPoint]
        :param normal: 境界線の向き
        :type normal: pygame.math.Vector2
        :param offset_point: 境界線が通る基準点
        :type offset_point: pygame.math.Vector2
        :param inc_edge_idx: 入射辺のインデックス
        :type inc_edge_idx: int
        :param offset_idx: 基準点のインデックス
        :type offset_idx: int
        :param inc_body: 入射オブジェクト
        :type inc_body: CollisionObject
        :param ref_body: 基準オブジェクト
        :type ref_body: CollisionObject
        :return: コンタクトポイントのリストを返す
        :rtype: list[ContactPoint]
        '''
        #入射面の頂点が二つ未満なら、即抜け
        if len(inc_face) < 2: return inc_face

        #クリップされた頂点リストを用意
        clipped_face:list[ContactPoint] = []

        #入射面を基準ベクトルに写像した時のオフセット点との距離を計算する。
        cp1, cp2 = inc_face[0], inc_face[1]
        dist_1 = normal.dot(cp1.point - offset_point)
        dist_2 = normal.dot(cp2.point - offset_point)

        # --dist_1,dist_2の値に基づいて条件分岐(境界線を跨ぐかどうか)--
        #v1が内側なら採用
        if dist_1 >= 0:
            clipped_face.append(cp1)

        #どちらかがオフセット点より小さいなら、入射面上の境界線の交点を計算し、リストに加える。
        if (dist_1 >= 0 and dist_2 < 0) or (dist_1 < 0 and dist_2 >= 0):
            #入射面上の点Pは(v1,P)と(P,v2)の比が、dist_1とdist_2の比に等しいことを利用して計算する。
            t = dist_1 / (dist_1 - dist_2)
            p = cp1.point + t * (cp2.point - cp1.point)
            #新しいコンタクトポイント作成、クリップされた点の場合、基準側の角と入射側の辺のペアとして扱う
            new_cp = ContactPoint(p)
            new_cp.generate_feature_id(ref_body,inc_body,('V', offset_idx),('E', inc_edge_idx))
            #頂点リストに追加
            clipped_face.append(new_cp)

        #同様にv2がオフセット点より大きいなら採用
        if dist_2 >= 0:
            clipped_face.append(cp2)
        
        # --- 小数点誤差によって発生する微小な重複点をインプレース（直接）で削除 ---
        # 削除によるインデックスのズレを防ぐため、後ろから順に確認する
        for i in range(len(clipped_face) - 1, 0, -1):
            # 自分と1つ手前の点の距離を比較
            if clipped_face[i].point.distance_squared_to(clipped_face[i-1].point) <= self.EPSILON_SQ:
                clipped_face.pop(i) # 重複していれば自分自身を削除
        
        return clipped_face

@dataclass
class ConstraintRow:
    '''
    物体間に働く拘束データを扱うデータクラスです。
    '''
    # --拘束の対象になる二つのオブジェクト--
    body_1:RigidBody2D = None 
    body_2:RigidBody2D = None 

    # --ヤコビアンJ--
    # --body_1側--
    linear_1:pygame.math.Vector2 = None # 物体１の並進成分
    angular_1:float = 0.0              # 物体１の回転成分(r×n)
    #--body_2側
    linear_2:pygame.math.Vector2 = None # 物体2の並進成分
    angular_2:float = 0.0              # 物体2の回転成分(r×n)

    effectiveMass:float = None #その拘束における物体の有効質量
    bias:float = None #その拘束が生じさせる内力の目標値
    accumulatedImpulse:float = 0.0 #拘束が発生させる力の実行値
    minImpulse:float = None # 拘束が発生させる力の最小値
    maxImpulse:float = None # 拘束が発生させる力の最大値

    #動的クランプ(例えば、摩擦が法線方向の力積に依存するような)の依存関係
    limit_reference:'ConstraintRow' = None #制限の基準となる別の拘束行
    limit_multiplier:float = 0.0  #基準に対する係数(摩擦係数等)

    def update_jacobian(self,r1:pygame.math.Vector2,r2:pygame.math.Vector2,direction:pygame.math.Vector2):
        '''
        ある方向(direction)とある地点(point)対するそれぞれの物体のヤコビアンを計算する。
        
        :param r1: 拘束地点の物体１を基準とした位置ベクトル
        :type r1: pygame.math.Vector2
        :param r2: 拘束地点の物体２を基準とした位置ベクトル
        :type r2: pygame.math.Vector2
        :param direction: 拘束の働く方向(物体１から２への方向)
        :type direction: pygame.math.Vector2
        '''
                #データの取得、整理
        b1,b2 = self.body_1,self.body_2

        #2D外積 (r x d) の計算 ,回転成分の計算
        cross_1 = r1.x * direction.y - r1.y * direction.x
        cross_2 = r2.x * direction.y - r2.y * direction.x

        # --ヤコビアンを構成--
        self.linear_1 = -direction
        self.angular_1 = -cross_1
        self.linear_2 = direction
        self.angular_2 = cross_2

        # --有効質量の計算--
        #並進のしやすさは質量の逆数の和
        inv_mass_sum = b1.inv_mass + b2.inv_mass

        #回転のしやすさ(r × n)^2 / I
        angular_term_1 = (cross_1**2) * b1.inv_inertia
        angular_term_2 = (cross_2**2) * b2.inv_inertia

        #合計して、有効質量の逆数を計算
        inv_effective_mass = inv_mass_sum + angular_term_1 + angular_term_2
        
        #行データに有効質量を保存、０除算回避
        self.effectiveMass = 1.0/inv_effective_mass if inv_effective_mass > 1e-9 else 0.0

class ConstraintUnit:
    '''
    論理的な拘束の固まり。
    1つ以上の ConstraintRow を管理し、物理的な意味（接触、距離、回転など）を保持する。
    '''
    def __init__(self, feature_id:tuple):
        self.feature_id = feature_id #その頂点を表すid
        self.rows = [] #その点に含まれる拘束データ
    
    def prepare(self, *args, **kwargs):
        '''
        各行データごとのヤコビアンやバイアスを計算する。
        サブクラスでオーバーライドする
        '''
        pass

class ContactConstraint(ConstraintUnit):
    '''
    接触による物体の拘束を保持するクラスです。
    '''
    def __init__(self, feature_id:tuple):
        super().__init__(feature_id)
        #法線拘束行を生成
        self.normal_row = ConstraintRow()
        #法線拘束の値域を定義、剛体の場合、跳ね返す場合は上限がなく、引き寄せる力はゼロ
        self.normal_row.minImpulse = 0.0
        self.normal_row.maxImpulse = float('inf')
        #摩擦拘束行を生成
        self.friction_row = ConstraintRow()
        #接触点が持つ拘束リスト
        self.rows = []

    def prepare(self, colisionresult:CollisionResult, contact_point:ContactPoint, dt:float, settings: PhysicsSettings):
        '''
        コリジョン判定の結果に基づき、その接触点(contact_point)における、ヤコビアンを生成します。
        
        :param colisionresult: コリジョン判定の結果が保存されています。(衝突している物体や法線ベクトルなど)
        :type colisionresult: CollisionResult
        :param contact_point: 接触点の座標
        :type contact_point: ContactPoint
        :param dt: ステップ時間
        :type dt: float
        :param settings: シミュレーションの設定
        :type settings: PhysicsSettings
        '''
        # --データの読み込み、準備 --
        normal = colisionresult.normal
        overlap = contact_point.depth
        point = contact_point.point
        b1,b2 = colisionresult.body_origin,colisionresult.body_target
        #拘束行を宣言
        normal_row = self.normal_row
        normal_row.body_1, normal_row.body_2 = b1, b2

        # --接触点に対するそれぞれのオブジェクトのローカル座標(位置ベクトルに変換する。)
        r1 = point - b1.transform.pos
        r2 = point - b2.transform.pos

        # --ヤコビアンJの計算 --
        normal_row.update_jacobian(r1,r2,normal)

        # --拘束地点での相対速度を計算する。
        v1 = b1.get_velocity_at_offset(r1)
        v2 = b2.get_velocity_at_offset(r2)
        relative_velocity = v2 - v1

        # ---------------------------------------------------------
        # -- 　 法線拘束 (normal_row) の固有処理 --
        # ---------------------------------------------------------
        # 相対速度を法線方向に投影 (JV)
        rel_v_normal = relative_velocity.dot(normal)
        
        #位置補正バイアスを計算
        # めり込み(depth)が slop を超えた分だけ補正速度を生成
        bias_pos = (settings.baumgarte / dt) * max(0.0, overlap - settings.slop)

        #反発バイアスを計算
        # --反発係数 (e) の決定と閾値処理--
        #相対速度が小さい（置かれているだけの）場合は反発させない
        if rel_v_normal > -settings.restitution_threshold:
            e = 0.0
        else:
            e = b1.restitution * b2.restitution
        # 相対速度の法線成分を反転させて e をかける
        # （接近速度 v_rel_n が負なので、反発目標速度は正の値になる）
        bias_restitution = -e * rel_v_normal
        
        # --- ソルバー標準形式 (Jv + bias = 0) への変換 ---
        # 目標速度(target)を達成するためには、ソルバー側で jv - target = 0 となればよい。
        # つまり、bias = -target として渡すのが正解。
        normal_row.bias = -(bias_restitution + bias_pos)
        #法線拘束をリストに追加,キャッシュの関係上、上書きしてリセット
        self.rows = [normal_row]

        # --摩擦による拘束の計算 --
        #合成の摩擦係数がゼロでないなら、摩擦拘束を生成
        if b1.friction > 0 and b2.friction > 0:
            #摩擦方向の拘束を生成
            friction_row = self.friction_row
            #拘束を構成する物体は法線拘束と同じ
            friction_row.body_1, friction_row.body_2 = b1, b2
            #拘束の方向は、法線方向に直交するベクトル
            tangent = pygame.math.Vector2(-normal.y,normal.x)
            #ヤコビアンを生成
            friction_row.update_jacobian(r1,r2,tangent)
            #摩擦の場合バイアスは常に０
            friction_row.bias = 0.0

            #二つの物体の合成摩擦係数を計算
            mu = math.sqrt(b1.friction * b2.friction)

            #摩擦力の値域は、法線方向の力積と合成摩擦係数の積に依存することを定義
            friction_row.limit_reference = normal_row
            friction_row.limit_multiplier = mu

            #拘束を追加
            self.rows.append(friction_row)

class InteractionLink:
    '''
    物体間に働く全ての拘束点の拘束を管理するクラス
    '''
    def __init__(self, b1:CollisionObject, b2:CollisionObject):
        '''
        :param b1: 拘束を構成するオブジェクト１
        :type b1: CollisionObject
        :param b2: 拘束を構成するオブジェクト２
        :type b2: CollisionObject
        '''
        self.body_1 = b1
        self.body_2 = b2
        # 二つオブジェクト間に働く拘束
        self.units: dict[tuple, ConstraintUnit] = {}
        self.last_active_step = 0

    def update(self, result:CollisionResult, dt:float, settings: PhysicsSettings):
        '''
        衝突判定の結果に基づき、全ての衝突点に対して、拘束情報作成や更新を行います。
        
        :param result: 衝突判定の結果
        :type result: CollisionResult
        :param dt: ステップ時間
        :type dt: float
        :param settings: シミュレーションの設定
        :type settings: PhysicsSettings
        '''
        new_units = {}
        # --接触拘束を取得し、構成する処理 --
        for contact in result.contact_points:
            #接触点idのキャッシュが存在するかどうかを調べる。
            if contact.id in self.units:
                unit = self.units[contact.id]
            else:
                unit = ContactConstraint(contact.id)
            
            #拘束のパラメータを更新
            unit.prepare(result,contact,dt,settings)

            #拘束を扱う辞書を最新の状態に更新する
            new_units[contact.id] = unit
        #前フレーム拘束状態を破棄して、新しく更新する。
        self.units = new_units

class ConstraintSolver:
    '''
    Projected Gauss-Seidel (PGS) アルゴリズムを用いて、
    物体の速度に関する拘束（めり込み解消・反発・摩擦など）を解くクラス。
    '''
    @staticmethod
    def solve(links: dict[tuple, InteractionLink], settings: PhysicsSettings):
        '''
        与えられた全てのInteractionLink から拘束行を展開し、大域的に拘束を解きます。
        
        :param links: 現在フレームの有効な物体間の拘束と、物体ペアのIDの辞書
        :type links: dict[tuple, InteractionLink]
        :param settings: 物理シミュレーションの設定データ
        :type settings: PhysicsSettings
        '''

        # --全ての拘束行を平坦なリストに展開 --
        all_rows: list[ConstraintRow] = []
        for link in links.values():
            for unit in link.units.values():
                all_rows.extend(unit.rows)

        # -- ウォームスタート --
        #accumulatedImpulseの初期値に基づいて、物体に力積を与える。キャッシュが残っている場合は、前回の結果が、ない場合は何も与えない。
        for row in all_rows:
            ConstraintSolver._apply_warm_start(row)

        # -- PGS反復計算 (Velocity Iteration) --
        for _ in range(settings.PGS_iterations):
            for row in all_rows:
                ConstraintSolver._solve_row(row)
        
    @staticmethod
    def _apply_warm_start(row: ConstraintRow):
        '''
        キャッシュされたaccumulatedImpulseを初期速度として適用する。
        
        :param row: 拘束行
        :type row: ConstraintRow
        '''
        #引き継いだ力積を取得
        impulse = row.accumulatedImpulse
        #もし力積値がゼロなら、キャッシュがないので、リターン
        if impulse == 0.0:
            return
        # -- 物体に力積を与える処理 --
        #力積を質量で割ったものが速度変化量(Δv = J * λ/M)
        if row.body_1:
            row.body_1.apply_constraint_impulse(row.linear_1, row.angular_1, impulse)
        if row.body_2:
            row.body_2.apply_constraint_impulse(row.linear_2, row.angular_2, impulse)
    
    @staticmethod
    def _solve_row(row: ConstraintRow):
        '''
       一つの拘束行に対して、目標値との偏差に応じて、力積を加えて速度を修正する。
        
        :param row: 拘束行
        :type row: ConstraintRow
        '''
        b1, b2 = row.body_1, row.body_2

        #摩擦など物体が他の拘束のパラメータに依存する場合の処理
        if row.limit_reference is not None:
            #値域は、参照先の力積と係数の積
            limit = row.limit_multiplier * row.limit_reference.accumulatedImpulse
            row.minImpulse = -limit
            row.maxImpulse = limit

        # --現在の相対速度を計算--
        jv = 0.0
        #それぞれの物体に対して、その拘束点における有効な速度を取得し、jvに加算する。
        if b1:
            jv += b1.get_jacobian_velocity(row.linear_1,row.angular_1)
        if b2:
            jv += b2.get_jacobian_velocity(row.linear_2, row.angular_2)
        
        # --必要な力積の差分(Δλ)を計算 --
        delta_impulse = - row.effectiveMass * (jv + row.bias)

        # --累積力積を有効な範囲にクランプする --
        old_impulse = row.accumulatedImpulse
        new_impulse = max(row.minImpulse, min(old_impulse + delta_impulse, row.maxImpulse))
        
        #クランプされた結果に基づいて、デルタインパルスを計算し直す(クランプされた場合実態と異なるため)
        delta_impulse = new_impulse - old_impulse
        row.accumulatedImpulse = new_impulse

        # --速度の更新 --
        if delta_impulse != 0.0:
            if b1:
                b1.apply_constraint_impulse(row.linear_1, row.angular_1, delta_impulse)
            if b2:
                b2.apply_constraint_impulse(row.linear_2, row.angular_2, delta_impulse)

class World:
    '''
    力学的な挙動のオブジェクトを統括して制御するワールドクラス
    '''
    def __init__(self,settings:PhysicsSettings):
        '''
        :param settings: 物理シミュレーションの設定
        :type settings: PhysicsSettings
        '''
        #ワールドクラスが管理する領域を定義
        self.width, self.height = settings.world_width, settings.world_height
        self.pos = pygame.math.Vector2(self.width/2,self.height/2)
        self.world_region = Region(self.pos, self.width/2, self.height/2)
        #4分木を管理するマネジャーを作成
        self.morton = MortonManager(self.world_region,settings.morton_depth)
        #ワールドクラスが制御する、オブジェクトのリスト
        self.bodies:list[CollisionObject] = []
        #シミュレーション設定を保持
        self.settings = settings
        #衝突判定マネージャーを保持
        self.collision_manager = CollisionManager()
        #重力を保持
        self.gravity = pygame.math.Vector2(settings.gravity)
        #拘束が働く物体リンクのキャッシュ情報{id:InteractionLink}
        self.Links_cash:dict[tuple,InteractionLink] = dict()

    def step(self, dt:float):
        '''
        ワールドクラスに含まれるオブジェクトをdt分進めます。重力を適用し、位置を更新させ、衝突を判定する。
        
        :param dt: 進めるステップ時間
        :type dt: float
        '''
        #1フレームの時間をサブステップで分割する。
        sub_dt = dt / self.settings.sub_steps
        for _ in range(self.settings.sub_steps):
            # 1. 重力の適用
            for body in self.bodies:
                body:CollisionObject
                if body.inv_mass > 0 and body.use_gravity:
                    body.apply_force(self.gravity * body.mass,sub_dt)

            # 2. 衝突解決
            self.resolve_collisions(sub_dt)

            # 3. 座標更新
            for body in self.bodies:
                body.update(sub_dt,self.settings)

        #4.消滅するオブジェクトの管理
        self.bodies = [b for b in self.bodies if not b._is_dead]

    def resolve_collisions(self,dt:float):
        '''
        衝突の解決処理を行います。また拘束状態のキャッシュを管理します。
        
        :param self: 説明
        :param dt: 説明
        :type dt: float
        '''
        #最新の拘束リンクの辞書
        new_links = dict()

        # --ブロードフェーズ： 空間分割マップを作成
        #モートンクラスの辞書を初期化
        self.morton.clear()
        #オブジェクトを分割された空間に登録する
        for body in self.bodies:
            self.morton.register(body)

        # --ナローフェーズ: オブジェクトの近辺のみを判定
        for body in self.bodies:
            #モートン空間から、衝突している可能性のあるオブジェクトのみを取得
            candidates = self.morton.get_collision_candidates(body)

            #すでに判定したペアを記録するセット
            checked_pairs = set()

            #オブジェクトと候補者との間で衝突判定する。
            for other in candidates:
                #二つの物体間の拘束を表すキーを作成、入力順によらず、IDは一定になる。
                Link_ID = self._generate_Ineraction_id(body, other)
                
                #リンクIDがすでに判定済みならスキップ
                if Link_ID in checked_pairs:
                    continue
                checked_pairs.add(Link_ID)

                #質量が０同士の判定はスキップする。
                if body.inv_mass == 0 and other.inv_mass == 0:
                    continue
                
                #コリジョン判定を行う。
                result = self.collision_manager.solve(body,other)
                if result.collided:
                    #リンクをインタラクションリンクとして定義
                    Link:InteractionLink
                    #link_IDがキャッシュに存在するかどうかを調べる。
                    if Link_ID in self.Links_cash:
                        #キーが存在するなら、キャッシュを利用
                        Link = self.Links_cash[Link_ID]
                    else:
                        #ないなら、新たにインタラクションリンクを作成
                        Link = InteractionLink(body, other)
                    #物体間の拘束の状態を、衝突判定の結果に基づいて更新
                    Link.update(result,dt,self.settings)
                    #最新のリンク辞書に登録する。
                    new_links[Link_ID] = Link

        #拘束ソルバに拘束データを渡して処理させる。反復回数は、シミュレーション設定での値に依存する。
        ConstraintSolver.solve(new_links,self.settings)
        #キャッシュを最新の状態に更新
        self.Links_cash = new_links

    def _generate_Ineraction_id(self, body1:CollisionObject, body2:CollisionObject) -> tuple:
        '''
        インタラクションリンクをキャッシュし、以降のフレームで追跡するためのidを作成します。idはオブジェクトのidの大小で二つの物体の順序を決めるので、入力順によらず、一意に定まります。
        例 id(body1) < id(body2）なら、ID = tuple[id(body1), id(body2)]
        
        :param body1: オブジェクト１
        :type body1: CollisionObject
        :param body2: オブジェクト２
        :type body2: CollisionObject
        :return: 作成されたID
        :rtype: tuple
        '''
        id_1 = id(body1)
        id_2 = id(body2)
        if id_1 < id_2:
            return (id_1, id_2)
        else:
            return (id_2, id_1)
        
class Renderer:
    # --- クラス定数として階層ごとの色を定義 ---
    LEVEL_COLORS = [
        (255, 50,  50),   # Level 0: 赤 (一番大きな枠、ルート)
        (255, 165, 0),    # Level 1: オレンジ
        (255, 255, 0),    # Level 2: 黄色
        (50,  255, 50),   # Level 3: 緑
        (50,  255, 255),  # Level 4: 水色
        (50,  50,  255),  # Level 5: 青
        (128, 0,   128)   # Level 6以上: 紫
    ]
    @staticmethod
    def render(screen, bodies: list[CollisionObject], *, show_debug=True):
        '''
        ワールドオブジェクト内に配置された、オブジェクトを描画します。

        :param bodies: 描画したい、オブジェクトのリスト
        :type bodies: list[Rigitbody2D]
        :param show_debug: デバック用の表示を有効にできます。
        :type show_debug: bool
        :param debug_contacts: 接触点のリスト
        :type debug_contacts: list[pygame.math.Vector2]
        '''
        for body in bodies:
            #オブジェクトの頂点座標のグローバル座標を取得
            points:list[pygame.math.Vector2] = body.transform.world_vertices
            
            # 描画実行
            pygame.draw.polygon(screen, body.color, points)
            
            #デバッグ表示
            if show_debug:
                Renderer._draw_debug(screen, body, points)

    @staticmethod
    def _draw_debug(screen, body, points):
        pos:pygame.math.Vector2 = body.transform.pos 
        
        # 中心点の描画（intにキャストしてタプルにするのが最も安全）
        center = (int(pos.x), int(pos.y))
        pygame.draw.circle(screen, (255, 255, 255), center, 2)

        # 向きを示す線の描画
        if len(points) > 0:
            start_pos = (pos.x, pos.y)
            end_pos = (float(points[0][0]), float(points[0][1]))
            
            pygame.draw.line(screen, (255, 255, 0), start_pos, end_pos, 1)
        #リージョンを表示
        region = body.region
        
        # PygameのRect描画用に (x, y, width, height) を計算して整数にキャスト
        rect_x = int(region.left)
        rect_y = int(region.top)
        rect_w = int(region.half_width * 2)
        rect_h = int(region.half_height * 2)

        # 登録されている階層から色を決定 
        color = (0, 255, 0) # デフォルトは緑
        if body.addresses:
            level = body.addresses[0][0] # addressesは [(level, morton_id), ...] の形式
            color_index = min(level, len(Renderer.LEVEL_COLORS) - 1)
            color = Renderer.LEVEL_COLORS[color_index]
        
        # 緑色の細い線(太さ1)でAABBを描画
        pygame.draw.rect(screen, color, (rect_x, rect_y, rect_w, rect_h), 1)

    @staticmethod
    def render_stats(screen, clock, physics_ms, body_count):
        font = pygame.font.SysFont("Consolas", 16)
        
        fps = clock.get_fps()
        # 表示テキストのリスト
        stats = [
            f"FPS: {fps:.1f}",
            f"Physics: {physics_ms:.2f} ms",
            f"Bodies: {body_count}",
        ]
        
        for i, text in enumerate(stats):
            img = font.render(text, True, (0, 255, 0)) 
            screen.blit(img, (10, 10 + i * 20))

class DebugStats:
    def __init__(self):
        self.physics_times = []
        
    def add_time(self, ms):
        self.physics_times.append(ms)
        if len(self.physics_times) > 60:
            self.physics_times.pop(0)
            
    def get_avg(self):
        if not self.physics_times: return 0
        return sum(self.physics_times) / len(self.physics_times)

class KinematicsLogger:
    '''
    特定のRigidBody2Dの速度および角速度の変化を独立して監視し、別々の履歴として記録するクラス
    '''
    def __init__(self, target_body: RigidBody2D, log_name: str = "body"):
        self.target_body = target_body
        self.log_name = log_name
        
        # 履歴リストを完全に分離
        self.velocity_history = []
        self.angular_history = []
        
        self.start_time = time.perf_counter()
        
        # それぞれ独立したコールバックを登録
        self.target_body.velocity_observers.append(self.record_velocity)
        self.target_body.angular_velocity_observers.append(self.record_angular_velocity)

    def record_velocity(self):
        '''速度のSetterが呼ばれた時だけ発火する'''
        current_time = time.perf_counter() - self.start_time
        vel = self.target_body.velocity
        self.velocity_history.append((current_time, vel.x, vel.y))

    def record_angular_velocity(self):
        '''角速度のSetterが呼ばれた時だけ発火する'''
        current_time = time.perf_counter() - self.start_time
        ang_vel = self.target_body.angular_velocity
        self.angular_history.append((current_time, ang_vel))

    def export_to_csv(self):
        '''
        記録した履歴を2つの別々のCSVファイルとして出力する
        '''
        # 1. 並進速度の出力
        vel_filename = f"vel_log_{self.log_name}.csv"
        with open(vel_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time(s)", "Velocity_X", "Velocity_Y"])
            for row in self.velocity_history:
                writer.writerow(row)
                
        # 2. 角速度の出力
        ang_filename = f"ang_log_{self.log_name}.csv"
        with open(ang_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time(s)", "Angular_Velocity"])
            for row in self.angular_history:
                writer.writerow(row)
                
        print(f"[{self.log_name}] ログ出力完了: 速度({len(self.velocity_history)}件), 角速度({len(self.angular_history)}件)")

if __name__ == '__main__':
    # --- メインループ部分 ---
    pygame.display.init() 
    pygame.font.init()  
    pygame.display.set_caption("Physics")
    clock = pygame.time.Clock()
    settings = PhysicsSettings(
        slop=0.1,
        baumgarte=0.005,
        restitution_threshold=50,
        PGS_iterations=10,
        velocity_threshold=0.01,
        gravity=(0,980))
    screen = pygame.display.set_mode((settings.world_width, settings.world_height))
    world = World(settings)

    # 地面を作成 (mass=0で固定, 色を指定)
    ground = RectBody(scale_x=800, scale_y=40, pos=(400, 580), restitution=0.8, mass=0.0, color=(50, 150, 50))
    world.bodies.append(ground)

    # 左右の壁
    left_wall = RectBody(scale_x=40, scale_y=600, pos=(20, 300), restitution=0.8, mass=0.0, color=(70, 70, 70))
    right_wall = RectBody(scale_x=40, scale_y=600, pos=(780, 300), restitution=0.8, mass=0.0, color=(70, 70, 70))
    world.bodies.append(left_wall)
    world.bodies.append(right_wall)

    #天井
    ceiling = RectBody(scale_x=800, scale_y=40, pos=(400,0),restitution=0.8,mass=0.0,color=(50,150,50))
    world.bodies.append(ceiling)

    # 中央に浮かぶブロック（少し回転させてみる）
    block = RectBody(scale_x=400, scale_y=30, pos=(400, 300), restitution=0.8,mass=0.0, angle=np.radians(10), color=(100, 100, 150))
    world.bodies.append(block)

    debug_stats = DebugStats()
    while True:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            # 左クリックでボール（CircleBody）を生成
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x_speed = random.randint(-400, 400)
                rand_color = (random.randint(150, 255), random.randint(50, 150), random.randint(50, 150))
                new_ball = CircleBody(radius=15, pos=event.pos, restitution=0.8, lifetime=100, color=rand_color,linear_damping=0.1,angular_damping=0.1)
                new_ball.velocity = pygame.math.Vector2(x_speed, 0) 
                world.bodies.append(new_ball)
                
            # 右クリックでランダムな四角形（RectBody）を生成
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                w, h = random.randint(20, 60), random.randint(20, 60)
                #rand_angle = np.radians(random.randint(0, 360))
                rand_color = (random.randint(50, 150), random.randint(150, 255), random.randint(200, 255))
                new_rect = RectBody(scale_x=40, scale_y=40,pos=event.pos, restitution=0.5, mass=10.0, color=rand_color,lifetime=100,friction=0.1)
                world.bodies.append(new_rect)
        
        # 物理演算の更新
        start_time = time.perf_counter()
        world.step(dt)
        end_time = time.perf_counter()

        # 描画処理
        screen.fill((30, 30, 30))
        
        # Rendererを呼び出す
        world.morton.draw_grid(screen)
        Renderer.render(screen, world.bodies, show_debug=False)

        # 平均を計算して表示
        debug_stats.add_time((end_time - start_time) * 1000)
        Renderer.render_stats(screen, clock, debug_stats.get_avg(), len(world.bodies))
        
        pygame.display.flip()