pipeline

ray batch (cur_ray), objects + sky

1. hit with all objects

    1. a ray may cross multiple objects but only the first hit is kept
    2. what should be recorded temporary

2. for every ray

    1. if it hits an object, create new batch of rays (next_ray)

    1. if it hits a light, or hits the sky (reaches infinity), accumulate the result to the image under rendering

3. collect all next_ray and go back to step 1



create new batch of rays 需要信息：

- hit point、source point 已经由 ray 结构提供
- normal vector，用于表示 hit 到的那个点所在面的法向量，用于实现漫反射模型、镜面反射模型
- intensity modulation 需要额外记录，用于设置新 ray 的初始颜色
- hit_a_light 记录该光线是否到达光源







