#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    isampler3D uInput; //quantized input
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec4 size;
  vec2 scale;
  ivec2 zero_point;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.size.xyz))) {
    vec4 texel = texelFetch(uInput, pos, 0);
    imageStore(
        uOutput,
        pos,
        (uBlock.scale.x * (texel - uBlock.zero_point.x)));
  }
}
