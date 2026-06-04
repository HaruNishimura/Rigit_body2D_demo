#include <iostream>
#include <raylib.h>
#include <cmath>
#include <vector>
#include <functional>
#include <utility>
#include <tuple>
#include <cassert>         
#include <initializer_list> 
#include <atomic>
#include <cstdint>
#include <type_traits>
#include <algorithm>

namespace Physics{
    /**
     * @brief 2次元ベクトル構造体
     * 物理演算における位置、速度、加速度の計算に使用する。
     * @tparam T 数値型 (float, double, intなど)
     */
    template<typename T>
    struct Vector2
    {
        //ベクトルの許容誤差
        static constexpr T EPSILON    = static_cast<T>(1e-5);
        //許容誤差の二乗(sqrt演算を回避する場合に利用)
        static constexpr T EPSILON_SQ = static_cast<T>(1e-10);
        //ベクトルのx成分
        T x = 0; 
        //ベクトルのy成分
        T y = 0;
        //  --コンストラクタ定義--
        Vector2() = default;
        Vector2(T vx, T vy) : x(vx) , y(vy) {}
        /*
          --演算子オーバーロードの定義--
        */
        Vector2 operator+(const Vector2& other) const{
            return {x + other.x, y +  other.y};
        }
        Vector2 operator-(const Vector2& other) const{
            return {x - other.x, y - other.y};
        }
        Vector2 operator*(const T factor) const{
            return {x * factor, y * factor};
        }
        Vector2 operator/(const T factor) const{
            return {x / factor, y / factor};
        }
        Vector2& operator+=(const Vector2& other) {
            x += other.x;
            y += other.y;
            return *this;
        }
        Vector2& operator-=(const Vector2& other) {
            x -= other.x;
            y -= other.y;
            return *this;
        }
        Vector2& operator*=(T factor){
            x *= factor;
            y *= factor;
            return *this;
        }
        Vector2& operator/=(T factor){
            x /= factor;
            y /= factor;
            return *this;
        }
        bool operator==(const Vector2& other) const{
            if constexpr (std::is_floating_point_v<T>) {
                return std::abs(x - other.x) <= EPSILON && std::abs(y - other.y) <= EPSILON;
            }else{
                return x == other.x && y == other.y;
            }
        }
        //raylibのベクトル２と暗黙的に変換できるようにする。
        operator ::Vector2() const{
            return {(float)x, (float)y};
        }
        /*
          --その他の演算の定義(内積や正規化、ノルムなど)--
        */
        //自身のベクトルに任意の値をセットします。
        Vector2& set(T newX, T newY){
            x = newX;
            y = newY;
            return *this;
        }
        //引数に指定したベクトルと自身との内積を返します。
        T dot(const Vector2& other) const{
            return x * other.x + y * other.y;
        }
        //引数に指定したベクトルと自身との外積を返します。
        T cross(const Vector2& other) const{
            return x * other.y - y * other.x;
        }
        //自身のベクトルの二乗ノルムを返します。
        T norm_sq() const{
            return x*x + y*y;
        }
        //自身のベクトルのノルムを返します。
        T norm() const{
            return std::sqrt(norm_sq());
        }
        //自身のベクトルを正規化して返します。
        Vector2 normalized() const{
            T len_sq = norm_sq();
            if constexpr (std::is_floating_point_v<T>) {
                if (len_sq <= EPSILON_SQ) return {0,0};
            }else{
                if (len_sq == 0) return {0,0};
            }
            return *this / std::sqrt(len_sq);
        }
        //自身のベクトルの値をコンソールに表示します。
        void show() const{
            std::cout << x << ',' << y << std::endl;
        }
    };

    //演算子の前に他の変数型のパラメータを持つ場合の演算子オーバーロード
    template<typename T>
    Vector2<T> operator*(T factor, const Vector2<T>& vec){
        return vec * factor;
    }

    /**
     * @brief 形状情報(頂点行列)を保存するクラス
     * オブジェクトの形状管理し、インデックスが循環するようにする。
     * @tparam T 数値型 (float, double, intなど)
     */
    template<typename T>
    class Vertices
    {
    private:
        std::vector<Vector2<T>> _data;
    
    public:
        //  --コンストラクタ定義--
        //デフォルトコンストラクタ
        Vertices() = default;
        //サイズのみを指定するコンストラクタ
        Vertices(size_t n) : _data(n) {}
        //ベクトル列を直接代入するコンストラクタ
        Vertices(std::initializer_list<Vector2<T>> list) : _data(list) {}

        //  --演算子オーバーロード--
        //読み取り専用アクセス(constから呼び出される)
        const Vector2<T>& operator[](int i) const{
            //サイズが０ならエラー
            assert(!_data.empty() && "Vertices: Attempted to read empty data!");
            //サイズ型をint型にキャストして保存
            int size = static_cast<int>(_data.size());
            //配列のレンジに収まるならそのまま返す
            if (i >= 0 && i < size) {
                return _data[i];
            }
            //負の値やサイズ越えを自動でループし、循環させる。
            return _data[(i % size + size) % size];
        }
        //書き換え用アクセス
        Vector2<T>& operator[](int i){
            //サイズが０ならエラー
            assert(!_data.empty() && "Vertices: Attempted to read empty data!");
            int size = static_cast<int>(_data.size());
            if(i >= 0 && i < size) {
                return _data[i];
            }
            return _data[(i % size + size) % size];
        }
        
        //  --イテレータ対応--
        auto begin() { return _data.begin(); }
        auto end() { return _data.end(); }
        auto begin() const { return _data.begin(); }
        auto end() const { return _data.end(); }

        // --メンバ関数定義--
        // 辺ベクトルを取得する関数
        Vector2<T> get_edge(int i) const {
            return (*this)[i + 1] - (*this)[i];
        }

        //  --std::vector 互換機能--
        //要素数を変更する関数
        void resize(size_t n) { _data.resize(n);}
        //要素数を定義する関数
        void reserve(size_t n) { _data.reserve(n);}
        //要素数を返す関数
        size_t size() const { return _data.size();}
        //オブジェクトが空かどうかを返す関数
        bool empty() const { return _data.empty();}
        //要素を追加する関数
        void push_back(const Vector2<T>& v) { _data.push_back(v); }
    };    
    template<typename T>
    class Transform2D{
    public:
        using Callback = std::function<void()>;
    private:
        // --クラスパラメータの定義--

        //オブジェクトの重心位置ベクトル
        Vector2<T> _pos;
        //オブジェクトの回転
        T _angle;
        //オブジェクトのスケールベクトル
        Vector2<T> _scale;

        //オブジェクトが持つ形状(頂点)情報列
        const Vertices<T>* _local_vertices = nullptr;
        //オブジェクトの姿勢を反映した、ワールド形状情報列
        mutable Vertices<T> _world_vertices_cache;

        //オブジェクトの回転情報から得られる三角関数
        mutable T _cos = 1, _sin = 0;
        //三角関数の更新が必要かどうか示すフラグ
        mutable bool _trig_dirty = true;
        //ワールド形状情報の更新が必要かどうかのフラグ
        mutable bool _vertices_dirty = true;

        //  --プライベート関数の定義--
        //自身のメンバの三角関数を更新する関数
        void update_trig() const{
            if(_trig_dirty){
                _cos = std::cos(_angle);
                _sin = std::sin(_angle);
                _trig_dirty = false;
            }
        }

        //自身のワールド頂点座標行列を更新する関数
        void update_world_vertices() const{
            //三角関数を更新処理を挟むのでゲッターを介して取得。
            auto t = trig();
            T c = t.first;
            T s = t.second;
            //ローカル頂点を取得
            const auto& locals = *_local_vertices;

            //ローカル頂点を一つずつワールド頂点に変換する
            for (size_t i = 0; i < locals.size(); ++i){
                //スケールを適用
                T sx = locals[i].x * _scale.x;
                T sy = locals[i].y * _scale.y;

                //回転を適用
                T rx = sx * c - sy * s;
                T ry = sx * s + sy * c;

                //最後に位置を適用
                _world_vertices_cache[i].x = rx + _pos.x;
                _world_vertices_cache[i].y = ry + _pos.y;
            }

            //フラグを折る
            _vertices_dirty = false;
        }
    public:
        //  --コンストラクタ定義--
        //デフォルトコンストラクタ
        Transform2D() : _pos(0,0), _angle(0), _scale(1,1) {}
        //引数ありのコンストラクタ
        Transform2D(Vector2<T> pos, T angle, Vector2<T> scale) 
            : _pos(pos), _angle(angle), _scale(scale){}
        //  --ゲッター定義--
        //posゲッター
        const Vector2<T>& pos() const{ return _pos;}

        //angleゲッター
        T angle() const { return _angle;}

        //scaleゲッター
        const Vector2<T>& scale() const { return _scale; }

        //trigゲッター pairで返す
        std::pair<T, T> trig() const{
            update_trig();
            return { _cos, _sin};
        }

        //ワールド頂点行列ゲッター
        const Vertices<T>& world_vertices() const{
            //形状データがない場合は空の状態で返す。
            if (!_local_vertices) {
                return _world_vertices_cache; 
            }
            if _vertices_dirty {
                update_world_vertices();
            }
            return _world_vertices_cache;
        }

        //  --セッター定義--
        //ベクトルオブジェクトを入力するposセッター
        void set_pos(const Vector2<T>& vec){
            _pos = vec;
            _vertices_dirty = true;
        }
        //パラメータを直接入力するposセッター
        void set_pos(T x, T y){
            _pos.set(x,y);
            _vertices_dirty = true;
        }
        //angleセッター
        void set_angle(T angle){
            _angle = angle;
            _vertices_dirty = true;
            _trig_dirty = true;
        }
        //ベクトルオブジェクトを入力するscaleセッター
        void set_scale(const Vector2<T>& scale){
            _scale = scale;
            _vertices_dirty = true;
        }
        //パラメータを直接入力するscaleセッター
        void set_scale(T x, T y){
            _scale.set(x,y);
            _vertices_dirty = true;
        }
        //形状情報(ローカル頂点行列)を登録するセッター
        void set_local_vertices(const Vertices<T>& vertices){
            //アドレスを取得して、ポインタに登録する。
            _local_vertices = &vertices;

            //ローカル頂点のサイズとワールド頂点のキャッシュのサイズは等しい(ローカル頂点はポインタなので、アロー演算子)
            _world_vertices_cache.resize(_local_vertices->size());
            _vertices_dirty = true;
        }  
    };
    /**
     * @brief 軸並行な領域(AABB)を保持する構造体。主にブロードフェーズで使うことを目的とします。
     * 
     * @tparam T 
     */
    template<typename T>
    struct  AABB {
        Vector2<T> min; //左上位置ベクトル
        Vector2<T> max; //右下位置ベクトル

        //コンストラクタ定義
        AABB() = default;

        //セッター定義(値を直接上書きする)
        void set(T minX, T minY, T maxX, T maxY) {
            min.set(minX,minY);
            max.set(maxX,maxY);
        }

        // --ゲッター定義--

        //AABBの幅を取得するゲッター
        T width() const { return max.x - min.x; }
        //AABBの高さを取得するゲッター
        T height() const { return max.y - min.y; }
        //最大辺の長さを返すゲッター
        T max_size() const { return std::max(width(),height()); }
 
        //二つのAABBが重なるかどうかを判定します。
        bool overlap(const AABB& other) const {
            return (min.x <= other.max.x && max.x >= other.min.x) && 
                   (min.y <= other.max.y && max.y >= other.min.y);
        }
    };

    template<typename T>
    class Shape{
    public:
        //形状を分類するクラスを定義
        enum class Type {
            Circle,
            Polygon,
            Rect,
            Triangle
        };

        //仮想デストラクタ
        virtual ~Shape() = default;

        //自身の形状(type)を返す仮想関数
        virtual Type get_type() const = 0;

        //頂点列を返す関数
        virtual const Vertices<T>& get_local_vertices() const = 0;

        //面積を計算する関数
        //virtual T get_area(const Vector2<T>& scale) const = 0;

        //慣性モーメント係数を計算する関数
        virtual T get_inertia_coefficient(const Vector2<T>& scale) const = 0;

        //形状を覆うAABB領域を計算する。
        virtual void compute_aabb(const Transform2D<T>& transform, AABB<T>& aabb) const = 0;
    protected:
        //多角形からAABBを計算するデフォルト関数(形状によってはこれよりも最適な処理が存在する可能性がある)
        //計算量:O(N)
        static void compute_aabb_from_vertices(const Vertices<T>& vertices, AABB<T>& aabb) {
            if (vertices.empty()) return;

            //先頭の頂点を使って最小、最大値を初期化
            T min_x = vertices[0].x, max_x = min_x;
            T min_y = vertices[0].y, max_y = min_y;

            //すべての頂点を走査し、最大と最小のx,yを取得
            for (const auto& v : vertices){
                if(v.x < min_x) min_x = v.x; else if(v.x > max_x) max_x = v.x;
                if(v.y < min_y) min_y = v.y; else if(v.y > max_y) max_y = v.y;
            }
            //AABBを更新
            aabb.set(min_x,min_y,max_x,max_y);
        }
    };

    template<typename T>
    class Circle : public Shape<T>{
    public:
        //円を多角形近似するときの分割数を定義
        static constexpr int DEFAULT_SEGMNTS = 32;
    private:
        //すべての円形が共有する。単位円の多角形近似の頂点列
        static const Vertices<T> _unit_vertices;

    public:
        //デフォルトコンストラクタを定義
        Circle() = default;

        //形状タイプを返す関数
        typename Shape<T>::Type get_type() const override{
            return Shape<T>::Type::Circle;
        }
        
        //頂点列への参照返す関数
        const Vertices<T>& get_local_vertices() const override{
            return _unit_vertices;
        }

        //慣性モーメント係数の計算：円の場合 1/2 * r^2で求められる
        T get_inertia_coefficient(const Vector2<T>& scale) const override{
            //半径をスケールから取得xとyスケールは円では同一であるので、xだけを取得
            T r = scale.x;
            return (T)0.5 * r * r;
        }

        //AABBの計算処理、円の場合、位置とスケールからO(1)で計算できる。
        void compute_aabb(const Transform2D<T>& transform, AABB<T>& aabb) const override {
            const auto& pos = transform.pos();
            //スケールから半径を取得
            T r = transform.scale().x;
            //AABBを更新
            aabb.set(pos.x - r, pos.y - r, pos.x + r, pos.y + r);
        }
    };

    template<typename T>
    const Vertices<T> Circle<T>::_unit_vertices = []() {
        Vertices<T> v;
        //クラス内の定数を参照
        const int n = Circle<T>::DEFAULT_SEGMNTS;
        //要素数文のメモリを確保
        v.reserve(n);
        for (int i =0; i < n; ++i) {
            T theta = static_cast<T>(2.0 * M_PI * i / n);
            v.push_back({std::cos(theta), std::sin(theta) });
        }
        return v;
    } ();

    template<typename T>
    class Rect : public Shape<T> {
    private:
        //全インスタンスで共有する単位正方形の頂点
        static const Vertices<T> _unit_vertices;
    
    public:
        //デフォルトコンストラクタを定義
        Rect() = default;

        //形状データを返すクラス
        typename Shape<T>::Type get_type() const override{
            return Shape<T>::Type::Rect;
        }

        //頂点列への参照を返す関数
        const Vertices<T>& get_local_vertices() const override{
            return _unit_vertices;
        }

        //慣性モーメント係数の計算
        //公式: 1/12 * (w^2 + h^2)
        T get_inertia_coefficient(const Vector2<T>& scale) const override{
            T w2 = scale.x * scale.x;
            T h2 = scale.y * scale.y;
            return (static_cast<T>(1.0) / static_cast<T>(12.0)) * (w2 + h2);
        }

        //AABBの更新処理、多角形用の汎用処理に渡す
        //計算量 : O(4)
        void compute_aabb(const Transform2D<T>& transform, AABB<T>& aabb) const override {
            this->compute_aabb_from_vertices(transform.world_vertices(),aabb);
        }
    };

    // --静的メンバの初期化 --
    template<typename T>
    const Vertices<T> Rect<T>::_unit_vertices = []() {
        Vertices<T> v;
        v.reserve(4);
        //中心(0,0)から、幅１、高さ１の正方形を反時計回りに定義
        v.push_back({static_cast<T>(-0.5), static_cast<T>(-0.5)});
        v.push_back({static_cast<T>( 0.5), static_cast<T>(-0.5)});
        v.push_back({static_cast<T>( 0.5), static_cast<T>( 0.5)});
        v.push_back({static_cast<T>(-0.5), static_cast<T>( 0.5)});
        return v;
    }();
    template<typename T>
    class Triangle : public Shape<T> {
    private:
        //三角形の基本頂点データ
        static const Vertices<T> _unit_vertices;
    public:
        //デフォルトコンストラクタを定義
        Triangle() = default;

        //形状を返す関数
        typename Shape<T>::Type get_type() const override {
            return Shape<T>::Type::Triangle;
        }

        //頂点列への参照を返す関数
        const Vertices<T>& get_local_vertices() const override {
            return _unit_vertices;
        }

        //慣性モーメント係数の計算
        //公式:1/6 * (a^2 / 4 + b^2 / 3)
        T get_inertia_coefficient(const Vector2<T>& scale) const override {
            T a2 = scale.x * scale.x;
            T b2 = scale.y * scale.y;
            return (static_cast<T>(1.0) / static_cast<T>(6.0)) * ((a2 / static_cast<T>(4.0)) + (b2 / static_cast<T>(3.0)));
        }

        //AABBの更新処理
        //計算量 : O(3)
        void compute_aabb(const Transform2D<T>& transform, AABB<T>& aabb) const override {
            this->compute_aabb_from_vertices(transform.world_vertices(), aabb);
        }
    };

    // --静的メンバの初期化--
    template<typename T>
    const Vertices<T> Triangle<T>::_unit_vertices = []() {
        Vertices<T> v;
        v.reserve(3);
        //底辺１、高さ１の二等辺三角形を定義
        T h = static_cast<T>(1.0);
        v.push_back({static_cast<T>(-0.5), static_cast<T>(-h / 3.0)});
        v.push_back({static_cast<T>( 0.5), static_cast<T>(-h / 3.0)});
        v.push_back({static_cast<T>( 0.0), static_cast<T>(2.0 * h / 3.0)});
        return v;
    }();
    /**
     * @brief 基本形状のシングルトンインスタンスを持つ構造体。基本形状を利用する場合、この構造体のインスタンスを参照して使う。
     * 
     * @tparam T 数値精度<float,double等>
     */
    template<typename T>
    struct DefaultShapes {
        //円形の基本形状
        static inline const Circle<T> circle;
        //矩形の基本形状
        static inline const Rect<T> rect;
        //二等辺三角形の基本形状
        static inline const Triangle<T> triangle;
    };
    
    /**
     * @brief 二次元剛体物理演算クラス : 質量や慣性モーメントといった力学的パラメータを管理するクラス
     * 
     * @tparam T 数値精度 (floatやdoubleを想定)
     * 
     */
    template<typename T>
    class Rigidbody2D{
    private:
        // --質量、慣性モーメント--
        //質量
        T _mass = 1;
        //逆質量
        T _inv_mass = 1;
        //慣性モーメント
        T _inertia = 1;
        //逆慣性モーメント
        T _inv_inertia = 1;

        // --物性的なパラメータ--
        //反発係数
        T _restitution = 0.8;
        //摩擦
        T _friction = 0.5;

        // --運動パラメータ--
        //速度
        Vector2<T> _velocity = {0,0};
        //角速度
        T _angular_velocity = 0;
        //力
        Vector2<T> _force = {0,0};
        //トルク
        T _torque = 0;

    public:
        // --セッター定義--
        //質量セッター
        void set_mass(T mass) {
            _mass = mass;
            _inv_mass = (mass > static_cast<T>(0) ? static_cast<T>(1.0) / mass 
                : static_cast<T>(0));
        }
        //慣性モーメントセッター
        void set_inertia(T inertia) {
            _inertia = inertia;
            _inv_inertia = (inertia > static_cast<T>(0) ? static_cast<T>(1.0) / inertia
                : static_cast<T>(0));
        }
        //反発係数セッター
        void set_restitution(T restitution) { _restitution = restitution; }
        //摩擦係数セッター
        void set_friction(T friction) { _friction = friction; }
        //速度セッター
        void set_velocity(const Vector2<T>& velocity) { _velocity = velocity; }
        //角速度セッター
        void set_angular_velocity(T angular_velocity) { _angular_velocity = angular_velocity; }

        //ゲッター定義
        //質量ゲッター
        T mass() const { return _mass; }
        //逆質量ゲッター
        T inv_mass() const { return _inv_mass; }
        //慣性モーメントゲッター
        T inertia() const { return _inertia; }
        //逆慣性モーメントゲッター
        T inv_inertia() const { return _inv_inertia; }
        //反発係数ゲッター
        T restitution() const { return _restitution; }
        //摩擦係数ゲッター
        T friction() const { return _friction; }
        //速度ゲッター
        const Vector2<T>& velocity() const { return _velocity; }
        //角速度ゲッター
        T angular_velocity() const { return _angular_velocity; }

        // --コンストラクタの定義--
        Rigidbody2D() = default;
        /**
         * @brief 剛体クラスのコンストラクタ
         * 
         * @param mass 質量
         * @param inertia 慣性モーメント
         * @param restitution 反発係数
         * @param friction 摩擦係数
         */
        Rigidbody2D(T mass, T inertia, T restitution, T friction) : 
             _restitution(restitution), _friction(friction) {
                set_mass(mass);
                set_inertia(inertia);
        }

        // --クラスメゾットの定義--
        //力を加えるメゾット
        void add_force(const Vector2<T>& force) { _force += force; }
        //トルクを加えるメゾット
        void add_torque(T torque) { _torque += torque; }
        /**
         * @brief セミインプリシット・オイラー法によって、
         *        力とトルクに基づいて速度、角速度を更新する関数。
         *        毎フレーム毎に呼び出されることを想定
         * 
         * @param dt ステップ時間
         */
        void integrate(T dt) {
            //質量無限なら固定
            if(_inv_mass == 0) return;

            //加速度計算し、単位時間との積を速度に加算する
            _velocity += (_force * _inv_mass) * dt;

            //慣性モーメント無限なら固定
            if (_inv_inertia == 0) return;

            //各加速度を計算し、単位時間との積を角速度に加算する
            _angular_velocity += (_torque * _inv_inertia) * dt;

            //トルク、力はステップごとに有効なので、初期化する
            _force = {0,0};
            _torque = 0;
        }
    };    
    template<typename T>
    class Entity{
    private:
        // --コンストラクタ定義--
        //基本的にファクトリメゾットからでしか生成を許可しない
        Entity() = default;

        // --パラメータ定義--
        Transform2D<T> _transform;
        Rigidbody2D<T> _rigidbody;
        //shapeは基底クラスであり、形の基本形は別の場所に実体を持つので、ポインタで定義
        const Shape<T>* _shape = nullptr;
        //一意なエンティティID
        uint32_t _id;
        //エンティティの形状を囲む、軸並行な領域AABB
        AABB<T> _aabb;

        /**
         * @brief エンティティを生成する関数
         * 
         * @param shape 形状情報へのポインタ
         * @param pos 位置ベクトル
         * @param scale スケールベクトル
         * @param mass 質量
         * @param restitution 反発係数 
         * @param friction 摩擦係数
         */
        void init(const Shape<T>* shape, const Vector2<T>& pos, const Vector2<T>& scale, T mass = 1.0, T restitution = 0.8, T friction = 0.5){
            //マルチスレッドで処理しても重複しないIDを発行
            static std::atomic<uint64_t> id_counter{1};
            //オブジェクトのIDを設定し、カウントを増やす
            _id = id_counter.fetch_add(1);
            //形状を設定
            _shape = shape;
            //姿勢の初期化
            _transform.set_pos(pos);
            _transform.set_scale(scale);
            //形状からローカル頂点を取得
            if (shape) {
                _transform.set_local_vertices(shape->get_local_vertices());
            }
            //慣性モーメントは一旦０で初期化する。
            T inertia = 0;
            //剛体の初期化
            _rigidbody = Rigidbody2D<T>(mass,inertia,restitution,friction);
            //慣性モーメントを係数する
            update_inertia();
            //AABBを計算する
            update_aabb();
        }

        //すでに実体のあるエンティティに対して、スケールや質量が変化した場合に慣性モーメントの更新処理を行う。
        void update_inertia() {
            //慣性モーメントを0で初期化
            T inertia = 0;
            if (_shape && _rigidbody.mass() > 0) {
                //慣性モーメントは質量とモーメント係数の積
                inertia = _rigidbody.mass() * _shape->get_inertia_coefficient(_transform.scale());
            }
            _rigidbody.set_inertia(inertia);
        }

        //AABBの更新処理、形状に応じた処理をShapeクラスに計算を委譲する
        void update_aabb() {
            if (_shape) {
                _shape->compute_aabb(_transform, _aabb);
            }
        }

    public:
        // --ゲッター定義--
        //エンティティのIDを返すゲッター
        uint64_t id() const { return _id; }
        //トランスフォーム読み取り参照
        const Transform2D<T>& transform() const { return _transform; }
        //リジッドボディ読み取り参照
        const Rigidbody2D<T>& rigidbody() const { return _rigidbody; }
        //シェイプ読み取り参照
        const Shape<T>* shape() const { return _shape; }
        //AABBゲッター
        const AABB<T>& aabb() const { return _aabb; }

        // --セッター定義--
        //リジットボディ書き込み可能参照
        Rigidbody2D<T>& rigidbody() { return _rigidbody; } 
        //位置ベクトルセッター
        //副作用:AABBの更新
        void set_pos(const Vector2<T>& new_pos) {
            _transform.set_pos(new_pos);
            update_aabb();
        }
        //位置ベクトルセッター
        //副作用：AABBの更新
        void set_pos(T x, T y) {
            _transform.set_pos(x,y);
            update_aabb();
        }
        //角度セッター
        //副作用：AABBの更新
        void set_angle(T new_angle) {
            _transform.set_angle(new_angle);
            update_aabb();
        }
        //スケールセッター
        //副作用：AABBの更新、慣性モーメントの更新
        void set_scale(const Vector2<T>& new_scale) {
            _transform.set_scale(new_scale);
            update_aabb();
            update_inertia();
        }
        //スケールセッター
        //副作用：AABBの更新、慣性モーメントの更新
        void set_scale(T x, T y) {
            _transform.set_scale(x,y);
            update_aabb();
            update_inertia();
        }
        //質量セッター
        //副作用：慣性モーメントの更新
        void set_mass(T mass) {
            _rigidbody.set_mass(mass);
            update_inertia();
        }

        // ======================
        // --パブリックメゾット定義--
        // ======================
        
        /**
         * @brief エンティティの位置と回転を更新
         * 
         * @param dt デルタタイム
         */
        void update_physics(T dt) {
            //力から速度へ積分
            _rigidbody.integrate(dt);

            //速度・角速度を速度に基づいて計算する。
            _transform.set_pos(_transform.pos() + _rigidbody.velocity() * dt);
            _transform.set_angle(_transform.angle() + _rigidbody.angular_velocity() * dt);

            //位置角度が変化した可能性があるのでAABBを更新
            update_aabb();
        }

        // ======================
        // --ファクトリメゾット定義--
        // ======================

        /**
         * @brief 任意半径をもつ円形エンティティを生成するファクトリメゾット
         * 
         * @param radius 半径
         * @param pos 位置ベクトル
         * @param mass 質量
         * @param restitution 反発係数 
         * @param friction 摩擦係数
         * @return Entity<T> 生成されたエンティティ
         */
        static Entity<T> create_circle(T radius, const Vector2<T>& pos, T mass = 1.0, T restitution = 0.8, T friction = 0.5) {
            Entity<T> e;
            e.init(&DefaultShapes<T>::circle, pos, {radius,radius}, mass, restitution, friction);
            return e;
        }

        /**
         * @brief 任意スケールを持つ矩形エンティティを生成するファクトリメゾット
         * 
         * @param width 横幅
         * @param height 高さ
         * @param pos 位置ベクトル
         * @param mass 質量
         * @param restitution 反発係数 
         * @param friction 摩擦係数
         * @return Entity<T> 
         */
        static Entity<T> create_rect(T width, T height, const Vector2<T>& pos, T mass = 1.0, T restitution = 0.8, T friction = 0.5) {
            Entity<T> e;
            e.init(&DefaultShapes<T>::rect, pos, {width, height}, mass, restitution, friction);
            return e;
        }

        /**
         * @brief 任意スケールを持つ三角形エンティティを生成するファクトリメゾット
         * 
         * @param base 底辺の長さ
         * @param height 高さ
         * @param pos 位置ベクトル
         * @param mass 質量
         * @param resutitution 反発係数
         * @param friction 摩擦係数
         * @return Entity<T> 
         */
        static Entity<T> create_triangle(T base, T height, const Vector2<T>& pos, T mass = 1.0, T restitution = 0.8, T friction = 0.5) {
            Entity<T> e;
            e.init(&DefaultShapes<T>::triangle, pos, {base, height}, mass, restitution, friction);
            return e;
        }

        /**
         * @brief 任意形状を生成するファクトリメゾット
         * 
         * @param custom_shape 任意形状
         * @param pos 位置
         * @param scale スケール
         * @param mass 質量
         * @param restitution　反発係数 
         * @param friction 摩擦係数
         * @return Entity<T> 
         */
        static Entity<T> create(const Shape<T>* custom_shape, const Vector2<T>& pos, const Vector2<T>& scale, T mass = 1.0, T restitution = 0.8, T friction = 0.5) {
            Entity<T> e;
            e.init(custom_shape, pos, scale, mass, restitution, friction);
            return e;
        }    
    };
    /**
     * @brief エンティティのID、AABBとモートンキーを紐付ける構造体
     * 
     * @tparam T 
     */
    template<typename T>
    struct Morton_proxy
    {
        //エンティティが登録された分割空間のモートンキー
        uint64_t morton_key;
        //エンティティのID
        uint64_t entity_id;
        //エンティティが持つ軸並行領域。
        AABB<T> aabb;
        //同じ空間にいる次のプロキシのインデックス
        int32_t next;
    };

    template<typename T>
    class Morton_manager
    {
    private:
        //空間分割の最大の深さ
        const uint8_t _max_level;

        //空間に登録できる最大オブジェクト数
        const uint64_t _max_objects;

        //モートンキーとその空間に登録されたオブジェクトのリンクリストの始点インデックスのハッシュマップ
        //<階層,モートン番号> : リンクリストの開始点
        std::unordered_map<uint64_t, int32_t> _grid;

        //プロキシプール(十分な数を用意し、毎フレーム使い回す)
        //モートンプロキシの配列
        std::vector<Morton_proxy<T>> _proxy_pool;
        //このフレームで有効なプールの開始点インデックス
        int32_t _pool_index = 0;   
    public:
        // --コンストラクタ定義--
        /**
         * @brief モートンマネジャーのコンストラクタ
         * 
         * @param max_objects :登録できる最大オブジェクト数
         * @param max_level :分割空間の最大深さ
         */
        Morton_manager(uint64_t max_objects, uint8_t max_level)
            : _max_objects(max_objects), _max_level(max_level) 
        {
            //プロキシプールを最大オブジェクト数分のメモリを確保
            //エンティティは大きさに応じて最大４つの空間に同時にプロキシとして登録するので、４倍のサイズが必要。
            _proxy_pool.resize(max_objects*4);
            //空間ハッシュマップの初期化
            _grid.reserve(max_objects*4);
        }
        /**
         * @brief モートンキーを作成するメゾット 64ビットの内上位8ビットが階層情報、下位56ビットをモートン番号に割り当てる
         * 
         * @param level 階層情報
         * @param morton_id 階層におけるモートン番号
         * @return * const uint64_t 
         */
        uint64_t make_key(uint8_t level, uint64_t morton_id) const{
            // levelを64bitにキャストしてから56ビット左シフトし、morton_idとORをとる
            return (static_cast<uint64_t>(level) << 56) | (morton_id &  0x00FFFFFFFFFFFFFF);
        }

        //毎フレーム呼ばれるハッシュマップの初期化処理
        void clear() {
            _grid.clear();
            _pool_index = 0; //オブジェクトは捨てずにインデックスを０に戻す
        }
        
        /**
         * @brief エンティティを指定した階層のモートン番号の空間に登録する処理
         * 
         * @param entity_id 
         * @param aabb 
         * @param level 
         * @param morton_id 
         */
        void insert(uint32_t entity_id, const AABB<T>& aabb, uint8_t level, uint64_t morton_id) {
            //現在のインデックスが配列のサイズを超えないかを確認
            if (_pool_index >= _proxy_pool.size()) {
                return; 
            }
            //階層とモートン番号から、モートンキーを作成
            uint64_t morton_key = make_key(level, morton_id);

            //プールから、プロキシノードを一つ取り出す。現在のプールインデックスを代入したのちインクリメント
            int32_t node_idx = _pool_index++;

            //プロキシにAABBとそれを持つエンティティのID、モートンキーを登録
            _proxy_pool[node_idx].entity_id = entity_id;
            _proxy_pool[node_idx].aabb = aabb;
            _proxy_pool[node_idx].morton_key = morton_key;

            //--リンクリストの繋ぎ込み処理--
            //モートンキーがすでにハッシュマップに登録されているなら
            if (_grid.find(morton_key) != _grid.end()){
                //現在のプロキシのリンク先をすでに登録されたオブジェクトを指すようにする。
                _proxy_pool[node_idx].next = _grid[morton_key];
            }else{
                //すでに登録されたオブジェクトがいないなら、終端
                _proxy_pool[node_idx].next = -1;
            }
            
            //ハッシュマップを更新
            _grid[morton_key] = node_idx;
        }
    };
}

int main() {
     
}