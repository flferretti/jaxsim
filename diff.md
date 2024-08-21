# Explicit dependencies

|Dependency|Before|After|Change|Explicit|Environments|
|-|-|-|-|-|-|
|black|24.4.2|24.8.0|Minor Upgrade|true|test-cpu on *all platforms*<br/>test-gpu on linux-64|
|cuda-cupti|12.5.39|12.6.37|Minor Upgrade|true|{gpu, test-gpu} on linux-64|
|cuda-version|12.5|12.6|Minor Upgrade|true|{gpu, test-gpu} on linux-64|
|idyntree|12.3.2|12.4.0|Minor Upgrade|true|test-cpu on *all platforms*<br/>test-gpu on linux-64|
|lxml|5.2.2|5.3.0|Minor Upgrade|true|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|mujoco|3.1.6|3.2.0|Minor Upgrade|true|*all envs* on linux-64|
|mujoco|3.1.5|3.2.0|Minor Upgrade|true|{default, test-cpu} on osx-64|
|pip|24.0|24.2|Minor Upgrade|true|test-cpu on *all platforms*<br/>test-gpu on linux-64|
|pre-commit|3.7.1|3.8.0|Minor Upgrade|true|test-cpu on *all platforms*<br/>test-gpu on linux-64|
|pytest|8.2.2|8.3.2|Minor Upgrade|true|test-cpu on *all platforms*<br/>test-gpu on linux-64|
|robot_descriptions|1.10.0|1.12.0|Minor Upgrade|true|test-cpu on *all platforms*<br/>test-gpu on linux-64|
|sdformat14|14.4.0|14.5.0|Minor Upgrade|true|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|jax|0.4.27|0.4.31|Patch Upgrade|true|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|jax-dataclasses|1.6.0|1.6.1|Patch Upgrade|true|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|jaxlib|0.4.25|0.4.31|Patch Upgrade|true|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|python|3.12.4|3.12.5|Patch Upgrade|true|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|rod|pyh0f4e4df_1|pyhd8ed1ab_2|Only build string|true|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|

# Implicit dependencies

|Dependency|Before|After|Change|Explicit|Environments|
|-|-|-|-|-|-|
|libgl||1.7.0|Added|false|{test-cpu, test-gpu} on linux-64|
|libglvnd||1.7.0|Added|false|{test-cpu, test-gpu} on linux-64|
|libglx||1.7.0|Added|false|{test-cpu, test-gpu} on linux-64|
|xcb-util||0.4.1|Added|false|{test-cpu, test-gpu} on linux-64|
|xorg-libxxf86vm||1.1.5|Added|false|{test-cpu, test-gpu} on linux-64|
|attrs|23.2.0|24.2.0|Major Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|gcc_impl_linux-64|12.3.0|13.3.0|Major Upgrade|false|{gpu, test-gpu} on linux-64|
|gcc_linux-64|12.3.0|13.3.0|Major Upgrade|false|{gpu, test-gpu} on linux-64|
|gxx_impl_linux-64|12.3.0|13.3.0|Major Upgrade|false|{gpu, test-gpu} on linux-64|
|gxx_linux-64|12.3.0|13.3.0|Major Upgrade|false|{gpu, test-gpu} on linux-64|
|harfbuzz|8.5.0|9.0.0|Major Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|icu|73.2|75.1|Major Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libcxx|17.0.6|18.1.8|Major Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}|
|libgcc-devel_linux-64|12.3.0|13.3.0|Major Upgrade|false|{gpu, test-gpu} on linux-64|
|libgcc-ng|13.2.0|14.1.0|Major Upgrade|false|*all envs* on linux-64|
|libgfortran-ng|13.2.0|14.1.0|Major Upgrade|false|*all envs* on linux-64|
|libgfortran5|13.2.0|14.1.0|Major Upgrade|false|*all envs* on linux-64|
|libgomp|13.2.0|14.1.0|Major Upgrade|false|*all envs* on linux-64|
|libsanitizer|12.3.0|13.3.0|Major Upgrade|false|{gpu, test-gpu} on linux-64|
|libstdcxx-devel_linux-64|12.3.0|13.3.0|Major Upgrade|false|{gpu, test-gpu} on linux-64|
|libstdcxx-ng|13.2.0|14.1.0|Major Upgrade|false|*all envs* on linux-64|
|setuptools|70.1.1|72.2.0|Major Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|kernel-headers_linux-64[^2]|4.18.0|3.10.0|Major Downgrade|false|{gpu, test-gpu} on linux-64|
|c-ares|1.28.1|1.33.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|ca-certificates|2024.6.2|2024.7.4|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|certifi|2024.6.2|2024.7.4|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|cffi|1.16.0|1.17.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|cuda-cccl_linux-64|12.5.39|12.6.37|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-crt-dev_linux-64|12.5.40|12.6.20|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-crt-tools|12.5.40|12.6.20|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-cudart|12.5.39|12.6.37|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-cudart-dev|12.5.39|12.6.37|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-cudart-dev_linux-64|12.5.39|12.6.37|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-cudart-static|12.5.39|12.6.37|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-cudart-static_linux-64|12.5.39|12.6.37|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-cudart_linux-64|12.5.39|12.6.37|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-driver-dev_linux-64|12.5.39|12.6.37|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-nvcc|12.5.40|12.6.20|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-nvcc-dev_linux-64|12.5.40|12.6.20|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-nvcc-impl|12.5.40|12.6.20|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-nvcc-tools|12.5.40|12.6.20|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-nvcc_linux-64|12.5.40|12.6.20|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-nvrtc|12.5.40|12.6.20|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-nvtx|12.5.39|12.6.37|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-nvvm-dev_linux-64|12.5.40|12.6.20|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-nvvm-impl|12.5.40|12.6.20|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|cuda-nvvm-tools|12.5.40|12.6.20|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|gnutls|3.7.9|3.8.7|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|identify|2.5.36|2.6.0|Minor Upgrade|false|test-cpu on *all platforms*<br/>test-gpu on linux-64|
|importlib-metadata|8.0.0|8.4.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|importlib_metadata|8.0.0|8.4.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|ipython|8.25.0|8.26.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|jsonschema|4.22.0|4.23.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|jsonschema-with-format-nongpl|4.22.0|4.23.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libcublas|12.5.2.13|12.6.0.22|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|libcurl|8.8.0|8.9.1|Minor Upgrade|false|*all envs* on linux-64|
|libcusparse|12.4.1.24|12.5.2.23|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|libdeflate|1.20|1.21|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libhwloc|2.10.0|2.11.1|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libmujoco|3.1.6|3.2.0|Minor Upgrade|false|*all envs* on linux-64|
|libmujoco|3.1.5|3.2.0|Minor Upgrade|false|{default, test-cpu} on osx-64|
|libnvjitlink|12.5.40|12.6.20|Minor Upgrade|false|{gpu, test-gpu} on linux-64|
|libopenvino|2024.2.0|2024.3.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libopenvino-arm-cpu-plugin|2024.2.0|2024.3.0|Minor Upgrade|false|{default, test-cpu} on osx-arm64|
|libopenvino-auto-batch-plugin|2024.2.0|2024.3.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libopenvino-auto-plugin|2024.2.0|2024.3.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libopenvino-hetero-plugin|2024.2.0|2024.3.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libopenvino-intel-cpu-plugin|2024.2.0|2024.3.0|Minor Upgrade|false|*all envs* on linux-64<br/>{default, test-cpu} on osx-64|
|libopenvino-intel-gpu-plugin|2024.2.0|2024.3.0|Minor Upgrade|false|*all envs* on linux-64|
|libopenvino-intel-npu-plugin|2024.2.0|2024.3.0|Minor Upgrade|false|*all envs* on linux-64|
|libopenvino-ir-frontend|2024.2.0|2024.3.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libopenvino-onnx-frontend|2024.2.0|2024.3.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libopenvino-paddle-frontend|2024.2.0|2024.3.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libopenvino-pytorch-frontend|2024.2.0|2024.3.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libopenvino-tensorflow-frontend|2024.2.0|2024.3.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libopenvino-tensorflow-lite-frontend|2024.2.0|2024.3.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libsdformat14|14.4.0|14.5.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|matplotlib-base|3.8.4|3.9.2|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|mujoco-python|3.1.6|3.2.0|Minor Upgrade|false|*all envs* on linux-64|
|mujoco-python|3.1.5|3.2.0|Minor Upgrade|false|{default, test-cpu} on osx-64|
|mujoco-samples|3.1.6|3.2.0|Minor Upgrade|false|*all envs* on linux-64|
|mujoco-samples|3.1.5|3.2.0|Minor Upgrade|false|{default, test-cpu} on osx-64|
|mujoco-simulate|3.1.6|3.2.0|Minor Upgrade|false|*all envs* on linux-64|
|mujoco-simulate|3.1.5|3.2.0|Minor Upgrade|false|{default, test-cpu} on osx-64|
|pillow|10.3.0|10.4.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|pyzmq|26.0.3|26.1.1|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|rpds-py|0.18.1|0.20.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|ruby|3.2.2|3.3.3|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|sdformat14-python|14.4.0|14.5.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|webcolors|24.6.0|24.8.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|wheel|0.43.0|0.44.0|Minor Upgrade|false|test-cpu on *all platforms*<br/>test-gpu on linux-64|
|zipp|3.19.2|3.20.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|zstandard|0.22.0|0.23.0|Minor Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|sysroot_linux-64[^2]|2.28|2.17|Minor Downgrade|false|{gpu, test-gpu} on linux-64|
|debugpy|1.8.2|1.8.5|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|exceptiongroup|1.2.0|1.2.2|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|ffmpeg|7.0.1|7.0.2|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|fonttools|4.53.0|4.53.1|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|fsspec|2024.6.0|2024.6.1|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|importlib_resources|6.4.0|6.4.3|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|ipykernel|6.29.4|6.29.5|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|jupyter_server|2.14.1|2.14.2|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|jupyterlab|4.2.3|4.2.4|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|jupyterlab_server|2.27.2|2.27.3|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libass|0.17.1|0.17.3|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|libcufft|11.2.3.18|11.2.6.28|Patch Upgrade|false|{gpu, test-gpu} on linux-64|
|libcurand|10.3.6.39|10.3.7.37|Patch Upgrade|false|{gpu, test-gpu} on linux-64|
|libcusolver|11.6.2.40|11.6.4.38|Patch Upgrade|false|{gpu, test-gpu} on linux-64|
|libdrm|2.4.121|2.4.122|Patch Upgrade|false|*all envs* on linux-64|
|libglib|2.80.2|2.80.3|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|pure_eval|0.2.2|0.2.3|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|pyyaml|6.0.1|6.0.2|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|sdl2|2.30.2|2.30.5|Patch Upgrade|false|test-cpu on *all platforms*<br/>test-gpu on linux-64|
|snappy|1.2.0|1.2.1|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|svt-av1|2.1.0|2.1.2|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|tqdm|4.66.4|4.66.5|Patch Upgrade|false|test-cpu on *all platforms*<br/>test-gpu on linux-64|
|trimesh|4.4.1|4.4.7|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|tyro|0.8.5|0.8.8|Patch Upgrade|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|types-python-dateutil|2.9.0.20240316|2.9.0.20240821|Other|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|_sysroot_linux-64_curr_repodata_hack|h69a702a_14|h69a702a_16|Only build string|false|{gpu, test-gpu} on linux-64|
|binutils_linux-64|hb3c18ed_9|hb3c18ed_0|Only build string|false|{gpu, test-gpu} on linux-64|
|bzip2|h10d778d_5|hfdf4475_7|Only build string|false|{default, test-cpu} on osx-64|
|bzip2|h93a5062_5|h99b78c6_7|Only build string|false|{default, test-cpu} on osx-arm64|
|bzip2|hd590300_5|h4bc722e_7|Only build string|false|*all envs* on linux-64|
|cairo|hbb29018_2|hebfffa5_3|Only build string|false|*all envs* on linux-64|
|cairo|hc6c324b_2|hb4a6bf7_3|Only build string|false|{default, test-cpu} on osx-arm64|
|cairo|h9f650ed_2|h37bd5c4_3|Only build string|false|{default, test-cpu} on osx-64|
|gettext|h59595ed_2|he02047a_3|Only build string|false|*all envs* on linux-64|
|gettext|h5ff76d1_2|hdfe23c8_3|Only build string|false|{default, test-cpu} on osx-64|
|gettext|h8fbad5d_2|h8414b35_3|Only build string|false|{default, test-cpu} on osx-arm64|
|gettext-tools|h59595ed_2|he02047a_3|Only build string|false|*all envs* on linux-64|
|gettext-tools|h5ff76d1_2|hdfe23c8_3|Only build string|false|{default, test-cpu} on osx-64|
|gettext-tools|h8fbad5d_2|h8414b35_3|Only build string|false|{default, test-cpu} on osx-arm64|
|irrlicht|h1344824_4|hdfd4c6d_5|Only build string|false|test-cpu on osx-arm64|
|irrlicht|h2a6caf8_4|hcce6d95_5|Only build string|false|{test-cpu, test-gpu} on linux-64|
|irrlicht|h5bfa9a0_4|hc01355b_5|Only build string|false|test-cpu on osx-64|
|libabseil|cxx17_hc1bcbd7_0|cxx17_hf036a51_1|Only build string|false|{default, test-cpu} on osx-64|
|libabseil|cxx17_h59595ed_0|cxx17_he02047a_1|Only build string|false|*all envs* on linux-64|
|libabseil|cxx17_hebf3989_0|cxx17_h00cdb27_1|Only build string|false|{default, test-cpu} on osx-arm64|
|libasprintf|h661eb56_2|he8f35ee_3|Only build string|false|*all envs* on linux-64|
|libasprintf|h5ff76d1_2|hdfe23c8_3|Only build string|false|{default, test-cpu} on osx-64|
|libasprintf|h8fbad5d_2|h8414b35_3|Only build string|false|{default, test-cpu} on osx-arm64|
|libasprintf-devel|h661eb56_2|he8f35ee_3|Only build string|false|*all envs* on linux-64|
|libasprintf-devel|h5ff76d1_2|hdfe23c8_3|Only build string|false|{default, test-cpu} on osx-64|
|libasprintf-devel|h8fbad5d_2|h8414b35_3|Only build string|false|{default, test-cpu} on osx-arm64|
|libblas|22_osxarm64_openblas|23_osxarm64_openblas|Only build string|false|{default, test-cpu} on osx-arm64|
|libblas|22_linux64_openblas|23_linux64_openblas|Only build string|false|*all envs* on linux-64|
|libboost|h17eb2be_3|hf763ba5_5|Only build string|false|test-cpu on osx-arm64|
|libboost|h739af76_3|hcca3243_5|Only build string|false|test-cpu on osx-64|
|libboost|hba137d9_3|h0ccab89_5|Only build string|false|{test-cpu, test-gpu} on linux-64|
|libcblas|22_osxarm64_openblas|23_osxarm64_openblas|Only build string|false|{default, test-cpu} on osx-arm64|
|libcblas|22_linux64_openblas|23_linux64_openblas|Only build string|false|*all envs* on linux-64|
|libgcrypt|h4ab18f5_0|h4ab18f5_1|Only build string|false|{test-cpu, test-gpu} on linux-64|
|libgettextpo|h59595ed_2|he02047a_3|Only build string|false|*all envs* on linux-64|
|libgettextpo|h5ff76d1_2|hdfe23c8_3|Only build string|false|{default, test-cpu} on osx-64|
|libgettextpo|h8fbad5d_2|h8414b35_3|Only build string|false|{default, test-cpu} on osx-arm64|
|libgettextpo-devel|h59595ed_2|he02047a_3|Only build string|false|*all envs* on linux-64|
|libgettextpo-devel|h5ff76d1_2|hdfe23c8_3|Only build string|false|{default, test-cpu} on osx-64|
|libgettextpo-devel|h8fbad5d_2|h8414b35_3|Only build string|false|{default, test-cpu} on osx-arm64|
|libintl|h5ff76d1_2|hdfe23c8_3|Only build string|false|{default, test-cpu} on osx-64|
|libintl|h8fbad5d_2|h8414b35_3|Only build string|false|{default, test-cpu} on osx-arm64|
|libintl-devel|h5ff76d1_2|hdfe23c8_3|Only build string|false|{default, test-cpu} on osx-64|
|libintl-devel|h8fbad5d_2|h8414b35_3|Only build string|false|{default, test-cpu} on osx-arm64|
|liblapack|22_osxarm64_openblas|23_osxarm64_openblas|Only build string|false|{default, test-cpu} on osx-arm64|
|liblapack|22_linux64_openblas|23_linux64_openblas|Only build string|false|*all envs* on linux-64|
|libmicrohttpd|h97afed2_0|hbc5bc17_1|Only build string|false|*all envs* on linux-64|
|libopenblas|pthreads_h413a1c8_0|pthreads_hac2b453_1|Only build string|false|*all envs* on linux-64|
|libopenblas|openmp_hfef2a42_0|openmp_h8869122_1|Only build string|false|{default, test-cpu} on osx-64|
|libopenblas|openmp_h6c19121_0|openmp_h517c56d_1|Only build string|false|{default, test-cpu} on osx-arm64|
|libspral|h1b93dcb_1|h831f25b_3|Only build string|false|{test-cpu, test-gpu} on linux-64|
|libtiff|h07db509_3|hf8409c0_4|Only build string|false|{default, test-cpu} on osx-arm64|
|libtiff|h129831d_3|h603087a_4|Only build string|false|{default, test-cpu} on osx-64|
|libtiff|h1dd3fc0_3|h46a8edc_4|Only build string|false|*all envs* on linux-64|
|libxml2|h3e169fe_1|heaf3512_4|Only build string|false|{default, test-cpu} on osx-64|
|libxml2|hc051c1a_1|he7c6b58_4|Only build string|false|*all envs* on linux-64|
|libxml2|ha661575_1|h01dff8b_4|Only build string|false|{default, test-cpu} on osx-arm64|
|llvm-openmp|hde57baf_0|hde57baf_1|Only build string|false|{default, test-cpu} on osx-arm64|
|llvm-openmp|h15ab845_0|h15ab845_1|Only build string|false|{default, test-cpu} on osx-64|
|nccl|hbc370b7_0|hbc370b7_1|Only build string|false|{gpu, test-gpu} on linux-64|
|openssl|hfb2fe0b_1|hfb2fe0b_2|Only build string|false|{default, test-cpu} on osx-arm64|
|openssl|h87427d6_1|h87427d6_2|Only build string|false|{default, test-cpu} on osx-64|
|openssl|h4ab18f5_1|h4bc722e_2|Only build string|false|*all envs* on linux-64|
|pcre2|h0f59acf_0|hba22ea6_2|Only build string|false|*all envs* on linux-64|
|pcre2|h7634a1b_0|h7634a1b_2|Only build string|false|{default, test-cpu} on osx-64|
|pcre2|h297a79d_0|h297a79d_2|Only build string|false|{default, test-cpu} on osx-arm64|
|python_abi|4_cp312|5_cp312|Only build string|false|{default, test-cpu} on {osx-64, osx-arm64}<br/>*all envs* on linux-64|
|qhull|h4bd325d_2|h434a139_5|Only build string|false|*all envs* on linux-64|
|qhull|hc021e02_2|h420ef59_5|Only build string|false|{default, test-cpu} on osx-arm64|
|qhull|h940c156_2|h3c5361c_5|Only build string|false|{default, test-cpu} on osx-64|
|scipy|py312hb9702fa_0|py312hb9702fa_2|Only build string|false|{default, test-cpu} on osx-64|
|scipy|py312hc2bc53b_0|py312h499d17b_2|Only build string|false|*all envs* on linux-64|
|scipy|py312h14ffa8f_0|py312h14ffa8f_2|Only build string|false|{default, test-cpu} on osx-arm64|
|tbb|h297d8ca_1|h434a139_3|Only build string|false|*all envs* on linux-64|
|tbb|h420ef59_1|h420ef59_3|Only build string|false|{default, test-cpu} on osx-arm64|
|tbb|h3c5361c_1|h3c5361c_3|Only build string|false|{default, test-cpu} on osx-64|

[^1]: **Bold** means explicit dependency.
[^2]: Dependency got downgraded.

