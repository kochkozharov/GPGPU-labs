/**
 * Ray Tracing на GPU (CUDA)
 * Вариант 3: Тетраэдр, Гексаэдр, Икосаэдр
 * На оценку 3: без рекурсии, без текстур, без отражений, один источник света
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

// ============================================================================
// Математические структуры и утилиты
// ============================================================================
// Используем встроенный float3 из CUDA (определен в vector_types.h)
// Добавляем операторы и утилиты для работы с ним

__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3& a, float t) {
    return make_float3(a.x * t, a.y * t, a.z * t);
}

__host__ __device__ inline float3 operator*(float t, const float3& a) {
    return make_float3(a.x * t, a.y * t, a.z * t);
}

__host__ __device__ inline float3 operator/(const float3& a, float t) {
    return make_float3(a.x / t, a.y / t, a.z / t);
}

__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline float3 normalize(const float3& v) {
    float len = length(v);
    if (len > 1e-6f) return v / len;
    return make_float3(0.0f, 0.0f, 0.0f);
}

// ============================================================================
// Структуры данных
// ============================================================================

struct Ray {
    float3 origin;
    float3 direction;
};

struct Triangle {
    float3 v0, v1, v2;
    float3 normal;
    int bodyIndex;  // к какому телу принадлежит
};

struct Body {
    float3 center;
    float3 color;      // нормированный цвет
    float radius;      // радиус описанной сферы
    float reflection;  // коэффициент отражения (не используется на 3)
    float transparency;// коэффициент прозрачности (не используется на 3)
    int edgeLights;    // источники на ребре (не используется на 3)
};

struct Light {
    float3 position;
    float3 color;
};

struct Camera {
    float3 position;
    float3 direction;
    float3 up;
    float3 right;
};

// ============================================================================
// Параметры камеры (цилиндрические координаты)
// ============================================================================

struct CameraParams {
    float r_c0, z_c0, phi_c0;
    float A_c_r, A_c_z;
    float omega_c_r, omega_c_z, omega_c_phi;
    float p_c_r, p_c_z;
    
    float r_n0, z_n0, phi_n0;
    float A_n_r, A_n_z;
    float omega_n_r, omega_n_z, omega_n_phi;
    float p_n_r, p_n_z;
};

// ============================================================================
// Глобальные константы для многогранников
// ============================================================================

// Тетраэдр: 4 вершины, 4 грани
#define TETRAHEDRON_VERTICES 4
#define TETRAHEDRON_FACES 4

// Гексаэдр (куб): 8 вершин, 6 граней (по 2 треугольника = 12 треугольников)
#define HEXAHEDRON_VERTICES 8
#define HEXAHEDRON_TRIANGLES 12

// Икосаэдр: 12 вершин, 20 граней
#define ICOSAHEDRON_VERTICES 12
#define ICOSAHEDRON_FACES 20

// Максимум треугольников для одного тела
#define MAX_TRIANGLES_PER_BODY 20

// ============================================================================
// Генерация вершин платоновых тел
// ============================================================================

__host__ void generateTetrahedron(float3 center, float radius, float3* vertices) {
    // Вершины правильного тетраэдра, вписанного в сферу радиуса radius
    float a = radius / sqrtf(3.0f);
    
    vertices[0] = make_float3(center.x + a, center.y + a, center.z + a);
    vertices[1] = make_float3(center.x + a, center.y - a, center.z - a);
    vertices[2] = make_float3(center.x - a, center.y + a, center.z - a);
    vertices[3] = make_float3(center.x - a, center.y - a, center.z + a);
}

__host__ void generateHexahedron(float3 center, float radius, float3* vertices) {
    // Куб, вписанный в сферу радиуса radius
    float a = radius / sqrtf(3.0f);
    
    vertices[0] = make_float3(center.x - a, center.y - a, center.z - a);
    vertices[1] = make_float3(center.x + a, center.y - a, center.z - a);
    vertices[2] = make_float3(center.x + a, center.y + a, center.z - a);
    vertices[3] = make_float3(center.x - a, center.y + a, center.z - a);
    vertices[4] = make_float3(center.x - a, center.y - a, center.z + a);
    vertices[5] = make_float3(center.x + a, center.y - a, center.z + a);
    vertices[6] = make_float3(center.x + a, center.y + a, center.z + a);
    vertices[7] = make_float3(center.x - a, center.y + a, center.z + a);
}

__host__ void generateIcosahedron(float3 center, float radius, float3* vertices) {
    // Икосаэдр, вписанный в сферу радиуса radius
    float phi = (1.0f + sqrtf(5.0f)) / 2.0f;  // золотое сечение
    float scale = radius / sqrtf(1.0f + phi * phi);
    
    // 12 вершин
    vertices[0]  = make_float3(center.x - scale, center.y + phi * scale, center.z);
    vertices[1]  = make_float3(center.x + scale, center.y + phi * scale, center.z);
    vertices[2]  = make_float3(center.x - scale, center.y - phi * scale, center.z);
    vertices[3]  = make_float3(center.x + scale, center.y - phi * scale, center.z);
    
    vertices[4]  = make_float3(center.x, center.y - scale, center.z + phi * scale);
    vertices[5]  = make_float3(center.x, center.y + scale, center.z + phi * scale);
    vertices[6]  = make_float3(center.x, center.y - scale, center.z - phi * scale);
    vertices[7]  = make_float3(center.x, center.y + scale, center.z - phi * scale);
    
    vertices[8]  = make_float3(center.x + phi * scale, center.y, center.z - scale);
    vertices[9]  = make_float3(center.x + phi * scale, center.y, center.z + scale);
    vertices[10] = make_float3(center.x - phi * scale, center.y, center.z - scale);
    vertices[11] = make_float3(center.x - phi * scale, center.y, center.z + scale);
}

// ============================================================================
// Генерация треугольников
// ============================================================================

__host__ int generateTetrahedronTriangles(float3* vertices, Triangle* triangles, int bodyIndex, float3 center) {
    // 4 грани тетраэдра
    int faces[4][3] = {
        {0, 1, 2},
        {0, 1, 3},
        {0, 2, 3},
        {1, 2, 3}
    };
    
    for (int i = 0; i < 4; i++) {
        triangles[i].v0 = vertices[faces[i][0]];
        triangles[i].v1 = vertices[faces[i][1]];
        triangles[i].v2 = vertices[faces[i][2]];
        float3 edge1 = triangles[i].v1 - triangles[i].v0;
        float3 edge2 = triangles[i].v2 - triangles[i].v0;
        float3 normal = normalize(cross(edge1, edge2));
        
        // Проверяем, направлена ли нормаль наружу от центра
        float3 faceCenter = (triangles[i].v0 + triangles[i].v1 + triangles[i].v2) / 3.0f;
        float3 outward = faceCenter - center;
        if (dot(normal, outward) < 0) {
            normal = normal * (-1.0f);  // Инвертируем нормаль
        }
        
        triangles[i].normal = normal;
        triangles[i].bodyIndex = bodyIndex;
    }
    return 4;
}

__host__ int generateHexahedronTriangles(float3* vertices, Triangle* triangles, int bodyIndex, float3 center) {
    // 6 граней, каждая из 2 треугольников
    int faces[12][3] = {
        // Нижняя грань (z = -a)
        {0, 1, 2}, {0, 2, 3},
        // Верхняя грань (z = +a)
        {4, 6, 5}, {4, 7, 6},
        // Передняя грань (y = -a)
        {0, 5, 1}, {0, 4, 5},
        // Задняя грань (y = +a)
        {2, 7, 3}, {2, 6, 7},
        // Левая грань (x = -a)
        {0, 7, 4}, {0, 3, 7},
        // Правая грань (x = +a)
        {1, 5, 6}, {1, 6, 2}
    };
    
    for (int i = 0; i < 12; i++) {
        triangles[i].v0 = vertices[faces[i][0]];
        triangles[i].v1 = vertices[faces[i][1]];
        triangles[i].v2 = vertices[faces[i][2]];
        float3 edge1 = triangles[i].v1 - triangles[i].v0;
        float3 edge2 = triangles[i].v2 - triangles[i].v0;
        float3 normal = normalize(cross(edge1, edge2));
        
        // Проверяем, направлена ли нормаль наружу от центра
        float3 faceCenter = (triangles[i].v0 + triangles[i].v1 + triangles[i].v2) / 3.0f;
        float3 outward = faceCenter - center;
        if (dot(normal, outward) < 0) {
            normal = normal * (-1.0f);  // Инвертируем нормаль
        }
        
        triangles[i].normal = normal;
        triangles[i].bodyIndex = bodyIndex;
    }
    return 12;
}

__host__ int generateIcosahedronTriangles(float3* vertices, Triangle* triangles, int bodyIndex, float3 center) {
    // 20 граней икосаэдра
    int faces[20][3] = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
    };
    
    for (int i = 0; i < 20; i++) {
        triangles[i].v0 = vertices[faces[i][0]];
        triangles[i].v1 = vertices[faces[i][1]];
        triangles[i].v2 = vertices[faces[i][2]];
        float3 edge1 = triangles[i].v1 - triangles[i].v0;
        float3 edge2 = triangles[i].v2 - triangles[i].v0;
        float3 normal = normalize(cross(edge1, edge2));
        
        // Проверяем, направлена ли нормаль наружу от центра
        float3 faceCenter = (triangles[i].v0 + triangles[i].v1 + triangles[i].v2) / 3.0f;
        float3 outward = faceCenter - center;
        if (dot(normal, outward) < 0) {
            normal = normal * (-1.0f);  // Инвертируем нормаль
        }
        
        triangles[i].normal = normal;
        triangles[i].bodyIndex = bodyIndex;
    }
    return 20;
}

// ============================================================================
// Вычисление позиции камеры
// ============================================================================

__host__ Camera computeCamera(const CameraParams& params, float t, int width, int height, float fovDeg) {
    Camera cam;
    
    // Позиция камеры в цилиндрических координатах
    float r_c = params.r_c0 + params.A_c_r * sinf(params.omega_c_r * t + params.p_c_r);
    float z_c = params.z_c0 + params.A_c_z * sinf(params.omega_c_z * t + params.p_c_z);
    float phi_c = params.phi_c0 + params.omega_c_phi * t;
    
    // Точка направления камеры
    float r_n = params.r_n0 + params.A_n_r * sinf(params.omega_n_r * t + params.p_n_r);
    float z_n = params.z_n0 + params.A_n_z * sinf(params.omega_n_z * t + params.p_n_z);
    float phi_n = params.phi_n0 + params.omega_n_phi * t;
    
    // Перевод в декартовы координаты
    cam.position = make_float3(r_c * cosf(phi_c), r_c * sinf(phi_c), z_c);
    float3 target = make_float3(r_n * cosf(phi_n), r_n * sinf(phi_n), z_n);
    
    cam.direction = normalize(target - cam.position);
    
    // Вектор "вверх" (ось Z)
    float3 worldUp = make_float3(0.0f, 0.0f, 1.0f);
    cam.right = normalize(cross(cam.direction, worldUp));
    cam.up = normalize(cross(cam.right, cam.direction));
    
    return cam;
}

// ============================================================================
// Пересечение луча с треугольником (алгоритм Моллера-Трумбора)
// ============================================================================

__device__ __host__ bool rayTriangleIntersect(
    const Ray& ray,
    const Triangle& tri,
    float& t,
    float3& hitNormal
) {
    const float EPSILON = 1e-6f;
    
    float3 edge1 = tri.v1 - tri.v0;
    float3 edge2 = tri.v2 - tri.v0;
    float3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);
    
    if (a > -EPSILON && a < EPSILON) return false;
    
    float f = 1.0f / a;
    float3 s = ray.origin - tri.v0;
    float u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f) return false;
    
    float3 q = cross(s, edge1);
    float v = f * dot(ray.direction, q);
    
    if (v < 0.0f || u + v > 1.0f) return false;
    
    t = f * dot(edge2, q);
    
    if (t > EPSILON) {
        hitNormal = tri.normal;
        return true;
    }
    
    return false;
}

// ============================================================================
// Пересечение луча с полом (плоскость z = floorZ)
// ============================================================================

__device__ __host__ bool rayFloorIntersect(
    const Ray& ray,
    float floorZ,
    float floorMinX, float floorMaxX,
    float floorMinY, float floorMaxY,
    float& t,
    float3& hitPoint
) {
    if (fabsf(ray.direction.z) < 1e-6f) return false;
    
    t = (floorZ - ray.origin.z) / ray.direction.z;
    if (t <= 0.0f) return false;
    
    hitPoint.x = ray.origin.x + t * ray.direction.x;
    hitPoint.y = ray.origin.y + t * ray.direction.y;
    hitPoint.z = floorZ;
    
    return (hitPoint.x >= floorMinX && hitPoint.x <= floorMaxX &&
            hitPoint.y >= floorMinY && hitPoint.y <= floorMaxY);
}

// ============================================================================
// CUDA Kernel для рендеринга
// ============================================================================

// Функция трассировки одного луча (device)
__device__ float3 traceRay(
    Ray ray,
    Triangle* triangles, int numTriangles,
    Body* bodies, int numBodies,
    Light* lights, int numLights,
    float floorZ, float floorMinX, float floorMaxX, float floorMinY, float floorMaxY,
    float3 floorColor
) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);  // Черный фон
    float minT = 1e30f;
    int hitBody = -1;
    float3 hitNormal;
    float3 hitPoint;
    bool hitFloor = false;
    
    // Проверка пересечения с телами
    for (int i = 0; i < numTriangles; i++) {
        float t;
        float3 normal;
        if (rayTriangleIntersect(ray, triangles[i], t, normal)) {
            if (t < minT) {
                minT = t;
                hitBody = triangles[i].bodyIndex;
                hitNormal = normal;
                hitPoint = ray.origin + ray.direction * t;
            }
        }
    }
    
    // Проверка пересечения с полом
    float floorT;
    float3 floorHit;
    if (rayFloorIntersect(ray, floorZ, floorMinX, floorMaxX, floorMinY, floorMaxY, floorT, floorHit)) {
        if (floorT < minT) {
            minT = floorT;
            hitBody = -2;
            hitNormal = make_float3(0.0f, 0.0f, 1.0f);
            hitPoint = floorHit;
            hitFloor = true;
        }
    }
    
    // Расчет освещения
    if (hitBody >= 0) {
        float3 bodyColor = bodies[hitBody].color;
        
        for (int i = 0; i < numLights; i++) {
            float3 lightDir = normalize(lights[i].position - hitPoint);
            float diff = fmaxf(0.0f, dot(hitNormal, lightDir));
            
            bool inShadow = false;
            Ray shadowRay;
            shadowRay.origin = hitPoint + hitNormal * 0.001f;
            shadowRay.direction = lightDir;
            float lightDist = length(lights[i].position - hitPoint);
            
            for (int j = 0; j < numTriangles && !inShadow; j++) {
                float t;
                float3 n;
                if (rayTriangleIntersect(shadowRay, triangles[j], t, n) && t < lightDist) {
                    inShadow = true;
                }
            }
            
            if (!inShadow) {
                color.x += bodyColor.x * lights[i].color.x * diff;
                color.y += bodyColor.y * lights[i].color.y * diff;
                color.z += bodyColor.z * lights[i].color.z * diff;
            }
        }
        
        color.x += bodyColor.x * 0.1f;
        color.y += bodyColor.y * 0.1f;
        color.z += bodyColor.z * 0.1f;
        
    } else if (hitFloor) {
        int cx = (int)floorf(hitPoint.x);
        int cy = (int)floorf(hitPoint.y);
        float checker = ((cx + cy) & 1) ? 1.0f : 0.5f;
        
        float3 baseColor = floorColor * checker;
        
        for (int i = 0; i < numLights; i++) {
            float3 lightDir = normalize(lights[i].position - hitPoint);
            float diff = fmaxf(0.0f, dot(hitNormal, lightDir));
            
            // Проверка тени на полу от объектов
            bool inShadow = false;
            Ray shadowRay;
            shadowRay.origin = hitPoint + hitNormal * 0.001f;
            shadowRay.direction = lightDir;
            float lightDist = length(lights[i].position - hitPoint);
            
            for (int j = 0; j < numTriangles && !inShadow; j++) {
                float t;
                float3 n;
                if (rayTriangleIntersect(shadowRay, triangles[j], t, n) && t < lightDist) {
                    inShadow = true;
                }
            }
            
            if (!inShadow) {
                color.x += baseColor.x * lights[i].color.x * diff;
                color.y += baseColor.y * lights[i].color.y * diff;
                color.z += baseColor.z * lights[i].color.z * diff;
            }
        }
        
        color.x += baseColor.x * 0.1f;
        color.y += baseColor.y * 0.1f;
        color.z += baseColor.z * 0.1f;
    }
    
    return color;
}

__global__ void renderKernel(
    unsigned char* output,
    int width, int height,
    float3 camPos, float3 camDir, float3 camUp, float3 camRight,
    float fovRad,
    Triangle* triangles, int numTriangles,
    Body* bodies, int numBodies,
    Light* lights, int numLights,
    float floorZ, float floorMinX, float floorMaxX, float floorMinY, float floorMaxY,
    float3 floorColor, float floorReflection,
    int ssaaSqrt,
    unsigned long long* rayCounter
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float aspect = (float)width / (float)height;
    float scale = tanf(fovRad / 2.0f);
    
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    int numSamples = ssaaSqrt * ssaaSqrt;
    
    // SSAA: выпускаем ssaaSqrt x ssaaSqrt лучей на пиксель
    for (int sy = 0; sy < ssaaSqrt; sy++) {
        for (int sx = 0; sx < ssaaSqrt; sx++) {
            // Подсчет лучей
            atomicAdd(rayCounter, 1ULL);
            
            // Субпиксельное смещение
            float subX = (sx + 0.5f) / ssaaSqrt;
            float subY = (sy + 0.5f) / ssaaSqrt;
            
            float px = (2.0f * ((x + subX) / width) - 1.0f) * aspect * scale;
            float py = (1.0f - 2.0f * ((y + subY) / height)) * scale;
            
            Ray ray;
            ray.origin = camPos;
            ray.direction = normalize(camDir + camRight * px + camUp * py);
            
            float3 sampleColor = traceRay(
                ray,
                triangles, numTriangles,
                bodies, numBodies,
                lights, numLights,
                floorZ, floorMinX, floorMaxX, floorMinY, floorMaxY,
                floorColor
            );
            
            color.x += sampleColor.x;
            color.y += sampleColor.y;
            color.z += sampleColor.z;
        }
    }
    
    // Усреднение
    color.x /= numSamples;
    color.y /= numSamples;
    color.z /= numSamples;
    
    // Clamp и запись в буфер
    color.x = fminf(1.0f, fmaxf(0.0f, color.x));
    color.y = fminf(1.0f, fmaxf(0.0f, color.y));
    color.z = fminf(1.0f, fmaxf(0.0f, color.z));
    
    int idx = (y * width + x) * 4;
    output[idx + 0] = (unsigned char)(color.x * 255.0f);
    output[idx + 1] = (unsigned char)(color.y * 255.0f);
    output[idx + 2] = (unsigned char)(color.z * 255.0f);
    output[idx + 3] = 255;  // Alpha (fully opaque)
}

// ============================================================================
// CPU версия рендеринга
// ============================================================================

// CPU версия трассировки луча
float3 traceRayCPU(
    Ray ray,
    Triangle* triangles, int numTriangles,
    Body* bodies, int numBodies,
    Light* lights, int numLights,
    float floorZ, float floorMinX, float floorMaxX, float floorMinY, float floorMaxY,
    float3 floorColor
) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float minT = 1e30f;
    int hitBody = -1;
    float3 hitNormal;
    float3 hitPoint;
    bool hitFloor = false;
    
    for (int i = 0; i < numTriangles; i++) {
        float t;
        float3 normal;
        if (rayTriangleIntersect(ray, triangles[i], t, normal)) {
            if (t < minT) {
                minT = t;
                hitBody = triangles[i].bodyIndex;
                hitNormal = normal;
                hitPoint = ray.origin + ray.direction * t;
            }
        }
    }
    
    float floorT;
    float3 floorHit;
    if (rayFloorIntersect(ray, floorZ, floorMinX, floorMaxX, floorMinY, floorMaxY, floorT, floorHit)) {
        if (floorT < minT) {
            minT = floorT;
            hitBody = -2;
            hitNormal = make_float3(0.0f, 0.0f, 1.0f);
            hitPoint = floorHit;
            hitFloor = true;
        }
    }
    
    if (hitBody >= 0) {
        float3 bodyColor = bodies[hitBody].color;
        
        for (int i = 0; i < numLights; i++) {
            float3 lightDir = normalize(lights[i].position - hitPoint);
            float diff = fmaxf(0.0f, dot(hitNormal, lightDir));
            
            bool inShadow = false;
            Ray shadowRay;
            shadowRay.origin = hitPoint + hitNormal * 0.001f;
            shadowRay.direction = lightDir;
            float lightDist = length(lights[i].position - hitPoint);
            
            for (int j = 0; j < numTriangles && !inShadow; j++) {
                float t;
                float3 n;
                if (rayTriangleIntersect(shadowRay, triangles[j], t, n) && t < lightDist) {
                    inShadow = true;
                }
            }
            
            if (!inShadow) {
                color.x += bodyColor.x * lights[i].color.x * diff;
                color.y += bodyColor.y * lights[i].color.y * diff;
                color.z += bodyColor.z * lights[i].color.z * diff;
            }
        }
        
        color.x += bodyColor.x * 0.1f;
        color.y += bodyColor.y * 0.1f;
        color.z += bodyColor.z * 0.1f;
        
    } else if (hitFloor) {
        int cx = (int)floorf(hitPoint.x);
        int cy = (int)floorf(hitPoint.y);
        float checker = ((cx + cy) & 1) ? 1.0f : 0.5f;
        
        float3 baseColor = floorColor * checker;
        
        for (int i = 0; i < numLights; i++) {
            float3 lightDir = normalize(lights[i].position - hitPoint);
            float diff = fmaxf(0.0f, dot(hitNormal, lightDir));
            
            // Проверка тени на полу от объектов
            bool inShadow = false;
            Ray shadowRay;
            shadowRay.origin = hitPoint + hitNormal * 0.001f;
            shadowRay.direction = lightDir;
            float lightDist = length(lights[i].position - hitPoint);
            
            for (int j = 0; j < numTriangles && !inShadow; j++) {
                float t;
                float3 n;
                if (rayTriangleIntersect(shadowRay, triangles[j], t, n) && t < lightDist) {
                    inShadow = true;
                }
            }
            
            if (!inShadow) {
                color.x += baseColor.x * lights[i].color.x * diff;
                color.y += baseColor.y * lights[i].color.y * diff;
                color.z += baseColor.z * lights[i].color.z * diff;
            }
        }
        
        color.x += baseColor.x * 0.1f;
        color.y += baseColor.y * 0.1f;
        color.z += baseColor.z * 0.1f;
    }
    
    return color;
}

void renderCPU(
    unsigned char* output,
    int width, int height,
    float3 camPos, float3 camDir, float3 camUp, float3 camRight,
    float fovRad,
    Triangle* triangles, int numTriangles,
    Body* bodies, int numBodies,
    Light* lights, int numLights,
    float floorZ, float floorMinX, float floorMaxX, float floorMinY, float floorMaxY,
    float3 floorColor, float floorReflection,
    int ssaaSqrt,
    unsigned long long& rayCount
) {
    float aspect = (float)width / (float)height;
    float scale = tanf(fovRad / 2.0f);
    int numSamples = ssaaSqrt * ssaaSqrt;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float3 color = make_float3(0.0f, 0.0f, 0.0f);
            
            // SSAA: выпускаем ssaaSqrt x ssaaSqrt лучей на пиксель
            for (int sy = 0; sy < ssaaSqrt; sy++) {
                for (int sx = 0; sx < ssaaSqrt; sx++) {
                    rayCount++;
                    
                    float subX = (sx + 0.5f) / ssaaSqrt;
                    float subY = (sy + 0.5f) / ssaaSqrt;
                    
                    float px = (2.0f * ((x + subX) / width) - 1.0f) * aspect * scale;
                    float py = (1.0f - 2.0f * ((y + subY) / height)) * scale;
                    
                    Ray ray;
                    ray.origin = camPos;
                    ray.direction = normalize(camDir + camRight * px + camUp * py);
                    
                    float3 sampleColor = traceRayCPU(
                        ray,
                        triangles, numTriangles,
                        bodies, numBodies,
                        lights, numLights,
                        floorZ, floorMinX, floorMaxX, floorMinY, floorMaxY,
                        floorColor
                    );
                    
                    color.x += sampleColor.x;
                    color.y += sampleColor.y;
                    color.z += sampleColor.z;
                }
            }
            
            // Усреднение
            color.x /= numSamples;
            color.y /= numSamples;
            color.z /= numSamples;
            
            color.x = fminf(1.0f, fmaxf(0.0f, color.x));
            color.y = fminf(1.0f, fmaxf(0.0f, color.y));
            color.z = fminf(1.0f, fmaxf(0.0f, color.z));
            
            int idx = (y * width + x) * 4;
            output[idx + 0] = (unsigned char)(color.x * 255.0f);
            output[idx + 1] = (unsigned char)(color.y * 255.0f);
            output[idx + 2] = (unsigned char)(color.z * 255.0f);
            output[idx + 3] = 255;
        }
    }
}

// ============================================================================
// Запись изображения в бинарном формате
// ============================================================================

bool writeImage(const std::string& path, int width, int height, const unsigned char* data) {
    std::ofstream fout(path, std::ios::binary);
    if (!fout) return false;
    
    fout.write(reinterpret_cast<const char*>(&width), sizeof(int));
    fout.write(reinterpret_cast<const char*>(&height), sizeof(int));
    fout.write(reinterpret_cast<const char*>(data), width * height * 4);
    
    return fout.good();
}

// ============================================================================
// Вывод конфигурации по умолчанию
// ============================================================================

void printDefaultConfig() {
    printf("120\n");                                // количество кадров
    printf("./output/frame_%%d.data\n");            // путь к изображениям
    printf("1280 720 90\n");                        // разрешение и FOV (HD для красивого результата)
    // Параметры камеры
    printf("8.0 3.0 0.0 2.0 1.5 1.0 3.0 1.0 0.0 0.0\n");  // r_c0, z_c0, phi_c0, A_c_r, A_c_z, omega_c_r, omega_c_z, omega_c_phi, p_c_r, p_c_z
    printf("0.5 0.5 0.0 0.2 0.3 0.5 1.5 1.0 0.0 0.0\n");  // r_n0, z_n0, phi_n0, A_n_r, A_n_z, omega_n_r, omega_n_z, omega_n_phi, p_n_r, p_n_z
    // Тетраэдр: центр, цвет, радиус, reflection, transparency, edge lights
    printf("2.5 0.0 1.0 1.0 0.3 0.3 1.0 0.0 0.0 0\n");
    // Гексаэдр
    printf("-2.0 2.5 0.8 0.3 1.0 0.3 0.9 0.0 0.0 0\n");
    // Икосаэдр
    printf("0.0 -2.5 1.2 0.3 0.3 1.0 1.1 0.0 0.0 0\n");
    // Пол: 4 точки, текстура (не используется), оттенок, reflection
    printf("-8.0 -8.0 -0.5 -8.0 8.0 -0.5 8.0 8.0 -0.5 8.0 -8.0 -0.5 none 0.9 0.9 0.9 0.0\n");
    // Источники света
    printf("1\n");
    printf("0.0 0.0 15.0 1.0 1.0 1.0\n");
    // Глубина рекурсии и SSAA (4x4 = 16 лучей на пиксель для сглаживания)
    printf("1 4\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    bool useGPU = true;
    bool printDefault = false;
    
    // Парсинг аргументов командной строки
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cpu") == 0) {
            useGPU = false;
        } else if (strcmp(argv[i], "--gpu") == 0) {
            useGPU = true;
        } else if (strcmp(argv[i], "--default") == 0) {
            printDefault = true;
        }
    }
    
    if (printDefault) {
        printDefaultConfig();
        return 0;
    }
    
    // Чтение параметров из stdin
    int numFrames;
    char outputPath[512];
    int width, height;
    float fovDeg;
    
    if (scanf("%d", &numFrames) != 1) {
        fprintf(stderr, "Error reading number of frames\n");
        return 1;
    }
    if (scanf("%s", outputPath) != 1) {
        fprintf(stderr, "Error reading output path\n");
        return 1;
    }
    if (scanf("%d %d %f", &width, &height, &fovDeg) != 3) {
        fprintf(stderr, "Error reading resolution and FOV\n");
        return 1;
    }
    
    // Параметры камеры
    CameraParams camParams;
    if (scanf("%f %f %f %f %f %f %f %f %f %f",
        &camParams.r_c0, &camParams.z_c0, &camParams.phi_c0,
        &camParams.A_c_r, &camParams.A_c_z,
        &camParams.omega_c_r, &camParams.omega_c_z, &camParams.omega_c_phi,
        &camParams.p_c_r, &camParams.p_c_z) != 10) {
        fprintf(stderr, "Error reading camera position parameters\n");
        return 1;
    }
    if (scanf("%f %f %f %f %f %f %f %f %f %f",
        &camParams.r_n0, &camParams.z_n0, &camParams.phi_n0,
        &camParams.A_n_r, &camParams.A_n_z,
        &camParams.omega_n_r, &camParams.omega_n_z, &camParams.omega_n_phi,
        &camParams.p_n_r, &camParams.p_n_z) != 10) {
        fprintf(stderr, "Error reading camera direction parameters\n");
        return 1;
    }
    
    // Тела (3 штуки для варианта 3)
    const int NUM_BODIES = 3;
    Body bodies[NUM_BODIES];
    
    for (int i = 0; i < NUM_BODIES; i++) {
        float cx, cy, cz;
        float r, g, b;
        float radius, reflection, transparency;
        int edgeLights;
        
        if (scanf("%f %f %f %f %f %f %f %f %f %d",
            &cx, &cy, &cz, &r, &g, &b, &radius, &reflection, &transparency, &edgeLights) != 10) {
            fprintf(stderr, "Error reading body %d parameters\n", i);
            return 1;
        }
        
        bodies[i].center = make_float3(cx, cy, cz);
        bodies[i].color = make_float3(r, g, b);
        bodies[i].radius = radius;
        bodies[i].reflection = reflection;
        bodies[i].transparency = transparency;
        bodies[i].edgeLights = edgeLights;
    }
    
    // Пол
    float floorPoints[12];
    char floorTexture[512];
    float floorTintR, floorTintG, floorTintB, floorReflection;
    
    if (scanf("%f %f %f %f %f %f %f %f %f %f %f %f %s %f %f %f %f",
        &floorPoints[0], &floorPoints[1], &floorPoints[2],
        &floorPoints[3], &floorPoints[4], &floorPoints[5],
        &floorPoints[6], &floorPoints[7], &floorPoints[8],
        &floorPoints[9], &floorPoints[10], &floorPoints[11],
        floorTexture,
        &floorTintR, &floorTintG, &floorTintB, &floorReflection) != 17) {
        fprintf(stderr, "Error reading floor parameters\n");
        return 1;
    }
    
    float floorZ = floorPoints[2];  // z координата пола
    float floorMinX = fminf(fminf(floorPoints[0], floorPoints[3]), fminf(floorPoints[6], floorPoints[9]));
    float floorMaxX = fmaxf(fmaxf(floorPoints[0], floorPoints[3]), fmaxf(floorPoints[6], floorPoints[9]));
    float floorMinY = fminf(fminf(floorPoints[1], floorPoints[4]), fminf(floorPoints[7], floorPoints[10]));
    float floorMaxY = fmaxf(fmaxf(floorPoints[1], floorPoints[4]), fmaxf(floorPoints[7], floorPoints[10]));
    float3 floorColor = make_float3(floorTintR, floorTintG, floorTintB);
    
    // Источники света
    int numLights;
    if (scanf("%d", &numLights) != 1) {
        fprintf(stderr, "Error reading number of lights\n");
        return 1;
    }
    
    std::vector<Light> lights(numLights);
    for (int i = 0; i < numLights; i++) {
        float px, py, pz, r, g, b;
        if (scanf("%f %f %f %f %f %f", &px, &py, &pz, &r, &g, &b) != 6) {
            fprintf(stderr, "Error reading light %d parameters\n", i);
            return 1;
        }
        lights[i].position = make_float3(px, py, pz);
        lights[i].color = make_float3(r, g, b);
    }
    
    // Глубина рекурсии и SSAA (не используется на оценку 3)
    int maxDepth, ssaaSqrt;
    if (scanf("%d %d", &maxDepth, &ssaaSqrt) != 2) {
        fprintf(stderr, "Error reading recursion depth and SSAA\n");
        return 1;
    }
    
    // Генерация треугольников для всех тел
    std::vector<Triangle> allTriangles;
    
    // Тетраэдр (тело 0)
    {
        float3 vertices[TETRAHEDRON_VERTICES];
        Triangle tris[TETRAHEDRON_FACES];
        generateTetrahedron(bodies[0].center, bodies[0].radius, vertices);
        int n = generateTetrahedronTriangles(vertices, tris, 0, bodies[0].center);
        for (int i = 0; i < n; i++) allTriangles.push_back(tris[i]);
    }
    
    // Гексаэдр (тело 1)
    {
        float3 vertices[HEXAHEDRON_VERTICES];
        Triangle tris[HEXAHEDRON_TRIANGLES];
        generateHexahedron(bodies[1].center, bodies[1].radius, vertices);
        int n = generateHexahedronTriangles(vertices, tris, 1, bodies[1].center);
        for (int i = 0; i < n; i++) allTriangles.push_back(tris[i]);
    }
    
    // Икосаэдр (тело 2)
    {
        float3 vertices[ICOSAHEDRON_VERTICES];
        Triangle tris[ICOSAHEDRON_FACES];
        generateIcosahedron(bodies[2].center, bodies[2].radius, vertices);
        int n = generateIcosahedronTriangles(vertices, tris, 2, bodies[2].center);
        for (int i = 0; i < n; i++) allTriangles.push_back(tris[i]);
    }
    
    int numTriangles = (int)allTriangles.size();
    
    // Выделение памяти
    size_t imageSize = width * height * 4;
    std::vector<unsigned char> hostImage(imageSize);
    
    float fovRad = fovDeg * 3.14159265f / 180.0f;
    
    if (useGPU) {
        // GPU рендеринг
        unsigned char* devImage;
        Triangle* devTriangles;
        Body* devBodies;
        Light* devLights;
        unsigned long long* devRayCounter;
        
        cudaMalloc(&devImage, imageSize);
        cudaMalloc(&devTriangles, numTriangles * sizeof(Triangle));
        cudaMalloc(&devBodies, NUM_BODIES * sizeof(Body));
        cudaMalloc(&devLights, numLights * sizeof(Light));
        cudaMalloc(&devRayCounter, sizeof(unsigned long long));
        
        cudaMemcpy(devTriangles, allTriangles.data(), numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
        cudaMemcpy(devBodies, bodies, NUM_BODIES * sizeof(Body), cudaMemcpyHostToDevice);
        cudaMemcpy(devLights, lights.data(), numLights * sizeof(Light), cudaMemcpyHostToDevice);
        
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                      (height + blockSize.y - 1) / blockSize.y);
        
        for (int frame = 0; frame < numFrames; frame++) {
            float t = (float)frame / (float)numFrames * 2.0f * 3.14159265f;
            
            Camera cam = computeCamera(camParams, t, width, height, fovDeg);
            
            unsigned long long zero = 0;
            cudaMemcpy(devRayCounter, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);
            
            auto startTime = std::chrono::high_resolution_clock::now();
            
            renderKernel<<<gridSize, blockSize>>>(
                devImage, width, height,
                cam.position, cam.direction, cam.up, cam.right,
                fovRad,
                devTriangles, numTriangles,
                devBodies, NUM_BODIES,
                devLights, numLights,
                floorZ, floorMinX, floorMaxX, floorMinY, floorMaxY,
                floorColor, floorReflection,
                ssaaSqrt,
                devRayCounter
            );
            
            cudaDeviceSynchronize();
            
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
            unsigned long long rayCount;
            cudaMemcpy(&rayCount, devRayCounter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            cudaMemcpy(hostImage.data(), devImage, imageSize, cudaMemcpyDeviceToHost);
            
            // Формирование пути к файлу
            char framePath[512];
            snprintf(framePath, sizeof(framePath), outputPath, frame);
            
            writeImage(framePath, width, height, hostImage.data());
            
            // Вывод статистики
            printf("%d\t%lld\t%llu\n", frame, (long long)duration.count(), rayCount);
        }
        
        cudaFree(devImage);
        cudaFree(devTriangles);
        cudaFree(devBodies);
        cudaFree(devLights);
        cudaFree(devRayCounter);
        
    } else {
        // CPU рендеринг
        for (int frame = 0; frame < numFrames; frame++) {
            float t = (float)frame / (float)numFrames * 2.0f * 3.14159265f;
            
            Camera cam = computeCamera(camParams, t, width, height, fovDeg);
            
            unsigned long long rayCount = 0;
            
            auto startTime = std::chrono::high_resolution_clock::now();
            
            renderCPU(
                hostImage.data(), width, height,
                cam.position, cam.direction, cam.up, cam.right,
                fovRad,
                allTriangles.data(), numTriangles,
                bodies, NUM_BODIES,
                lights.data(), numLights,
                floorZ, floorMinX, floorMaxX, floorMinY, floorMaxY,
                floorColor, floorReflection,
                ssaaSqrt,
                rayCount
            );
            
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
            char framePath[512];
            snprintf(framePath, sizeof(framePath), outputPath, frame);
            
            writeImage(framePath, width, height, hostImage.data());
            
            printf("%d\t%lld\t%llu\n", frame, (long long)duration.count(), rayCount);
        }
    }
    
    return 0;
}

