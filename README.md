
# RecSys ’25 Artifacts Evaluation

## GreenRec: Carbon-aware Recommendation System with Approximate Computation Reuse

## Contents
- [1. Constraints](#1-constraints)
- [2. Getting Started Instructions](#2-getting-started-instructions)
- [3. Kernel Build](#3-kernel-build)
- [4. NVMeVirt Build](#4-nvmevirt-build)
- [5. Conducting Evaluation](#5-conducting-evaluation)
- [6. Results](#6-results)

## 1. Constraints

The experimental environment is configured as follows:

| **Component** | **Specification**                  |
|---------------|------------------------------------|
| Processor     | Intel Xeon Gold 5218 2.3 GHz, 64-Core |
| Chipset       | Intel C621                         |
| Memory        | DDR4 2666 MHz, 128 GB (32 GB x4)  |
| GPU            | NVIDIA RTX 2090 |
| OS            | Ubuntu 20.04 Server (kernel v5.16.1)|

## 2. Getting Started Instructions

Retrieve the necessary code from GitHub by executing the script below:
```bash
./cloneall.sh
```
Upon completion, the modified kernel and NVMeVirt will be downloaded into their respective directories.

## 3. Kernel Build

Before building the kernel, ensure the following packages are installed:
```bash
apt-get update
apt-get install build-essential libncurses5 libncurses5-dev bin86 kernel-package libssl-dev bison flex libelf-dev dwarves
```
Configure the kernel:
```bash
cd kernel
make olddefconfig
```
To modify the `.config` file for building, replace the values of CONFIG_SYSTEM_TRUSTED_KEYS and CONFIG_SYSTEM_REVOCATION_KEYS with an empty string (""). This can be found near line 10477.

```bash
CONFIG_SYSTEM_TRUSTED_KEYS=""
CONFIG_SYSTEM_REVOCATION_KEYS=""
```

Build the kernel:
```bash
make -j$(nproc) LOCALVERSION=
sudo make INSTALL_MOD_STRIP=1 modules_install  
make install
```
Reboot and boot into the newly built kernel.

## 4. NVMeVirt Build

Verify the kernel version:
```bash
uname -r
```
Once "5.15.0DA_515" is confirmed, proceed as follows:
```bash
cd nvmevirt
./build.sh
```
After the build, `nvmev_on.ko` and `nvmev_off.ko` will be copied to the evaluation directory.

## 5. Conducting Evaluation

Reserve memory for the emulated NVMe device's storage by modifying `/etc/default/grub`:
```bash
GRUB_CMDLINE_LINUX="memmap=128G\\$256G intremap=off”
```
This reserves 128 GiB of physical memory starting at the 256 GiB offset. Adjust these values based on your physical memory size and storage capacity.

Update grub and reboot:
```bash
sudo update-grub
sudo reboot
```

[todo]

## 6. Results
*Details about evaluating and interpreting results.*
