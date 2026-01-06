"""
Quick GPU Detection Script
Run this before training to check if TensorFlow can use your GPU
"""

import tensorflow as tf
import sys

print("="*70)
print("GPU DETECTION TEST")
print("="*70)

# TensorFlow version
print(f"\nTensorFlow version: {tf.__version__}")

# Check if built with CUDA
print(f"Built with CUDA support: {tf.test.is_built_with_cuda()}")

# List all physical devices
print("\n" + "-"*70)
print("PHYSICAL DEVICES DETECTED:")
print("-"*70)

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

if len(gpus) > 0:
    print(f"\n✓ GPU DETECTED: {len(gpus)} GPU(s) available")
    for i, gpu in enumerate(gpus):
        print(f"  - GPU {i}: {gpu.name}")
        # Get GPU details
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            if gpu_details:
                print(f"    Device name: {gpu_details.get('device_name', 'Unknown')}")
        except:
            pass
else:
    print("\n✗ NO GPU DETECTED - Will use CPU only")
    print("  TensorFlow will run on CPU, which is MUCH slower for deep learning")

print(f"\nCPUs available: {len(cpus)}")
for i, cpu in enumerate(cpus):
    print(f"  - CPU {i}: {cpu.name}")

# Test computation
print("\n" + "-"*70)
print("RUNNING TEST COMPUTATION:")
print("-"*70)

try:
    if len(gpus) > 0:
        # Force GPU computation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            device_used = c.device
        print(f"\n✓ Test executed successfully on: {device_used}")
        if 'GPU' in device_used:
            print("\n✅ SUCCESS! Your laptop IS using GPU for TensorFlow operations")
        else:
            print("\n⚠ WARNING: Computation ran but not on GPU")
    else:
        # CPU only
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print(f"\n✓ Test executed on: {c.device}")
        print("\n⚠ Your laptop is using CPU only (no GPU detected)")
except Exception as e:
    print(f"\n✗ Error during test: {e}")

# Summary
print("\n" + "="*70)
print("SUMMARY:")
print("="*70)

if len(gpus) > 0:
    print("\n✅ GPU IS AVAILABLE AND READY TO USE")
    print("   Your training will be FAST (5-10x faster than CPU)")
    print("\n   Expected training time for 857 clients:")
    print("   - With GPU: ~2-4 hours")
    print("   - Without GPU: ~10-15 hours")
else:
    print("\n❌ NO GPU DETECTED - USING CPU ONLY")
    print("   Your training will be SLOW")
    print("\n   To enable GPU:")
    print("   1. Check if your laptop has an NVIDIA GPU")
    print("   2. Install CUDA Toolkit 11.8")
    print("   3. Install cuDNN 8.6")
    print("   4. Reinstall: pip install tensorflow[and-cuda]==2.13.0")
    print("\n   Expected training time for 857 clients:")
    print("   - CPU only: ~10-15 hours (or more)")

print("="*70)
