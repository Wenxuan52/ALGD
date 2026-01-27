# depends on the system and needs
# 下载 patchelf
curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf

# 给 patchelf 添加执行权限
chmod +x /usr/local/bin/patchelf

# 安装其他依赖包
apt update
apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev
