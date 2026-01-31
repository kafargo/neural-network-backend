#!/usr/bin/env python3
"""
Convert MNIST pickle file to modern NPZ format.

This script converts the legacy mnist.pkl.gz file (which triggers NumPy 2.4+
deprecation warnings) to a modern .npz format that is fully compatible with
current NumPy versions.

Usage:
    python scripts/convert_mnist_to_npz.py

The script will:
1. Load the existing mnist.pkl.gz file
2. Save it as mnist.npz in the same directory
3. Create a backup of the original file
4. Verify the conversion was successful
"""

import os
import sys
import pickle
import gzip
import shutil
import warnings
from typing import Tuple

import numpy as np


def load_legacy_mnist(filepath: str) -> Tuple:
    """
    Load MNIST data from legacy pickle format.

    Parameters:
    -----------
    filepath : str
        Path to the mnist.pkl.gz file

    Returns:
    --------
    tuple
        (training_data, validation_data, test_data)
        Each is a tuple of (images, labels)
    """
    print(f"📂 Loading legacy MNIST data from: {filepath}")

    # Suppress deprecation warnings during load
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        with gzip.open(filepath, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            training_data, validation_data, test_data = u.load()

    print(f"✅ Loaded successfully:")
    print(f"   - Training: {len(training_data[0])} images")
    print(f"   - Validation: {len(validation_data[0])} images")
    print(f"   - Test: {len(test_data[0])} images")

    return training_data, validation_data, test_data


def save_as_npz(data: Tuple, filepath: str) -> None:
    """
    Save MNIST data in modern NPZ format.

    Parameters:
    -----------
    data : tuple
        (training_data, validation_data, test_data)
    filepath : str
        Output path for the .npz file
    """
    print(f"\n💾 Converting to NPZ format: {filepath}")

    training_data, validation_data, test_data = data

    # Save as compressed NPZ
    np.savez_compressed(
        filepath,
        train_images=training_data[0],
        train_labels=training_data[1],
        val_images=validation_data[0],
        val_labels=validation_data[1],
        test_images=test_data[0],
        test_labels=test_data[1]
    )

    # Get file sizes for comparison
    npz_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
    print(f"✅ Saved successfully (size: {npz_size:.2f} MB)")


def verify_conversion(npz_filepath: str, original_data: Tuple) -> bool:
    """
    Verify that the NPZ file contains the same data as the original.

    Parameters:
    -----------
    npz_filepath : str
        Path to the .npz file
    original_data : tuple
        Original (training_data, validation_data, test_data)

    Returns:
    --------
    bool
        True if verification passes
    """
    print(f"\n🔍 Verifying conversion...")

    training_data, validation_data, test_data = original_data

    with np.load(npz_filepath) as data:
        # Check training data
        assert np.array_equal(data['train_images'], training_data[0]), \
            "Training images don't match!"
        assert np.array_equal(data['train_labels'], training_data[1]), \
            "Training labels don't match!"

        # Check validation data
        assert np.array_equal(data['val_images'], validation_data[0]), \
            "Validation images don't match!"
        assert np.array_equal(data['val_labels'], validation_data[1]), \
            "Validation labels don't match!"

        # Check test data
        assert np.array_equal(data['test_images'], test_data[0]), \
            "Test images don't match!"
        assert np.array_equal(data['test_labels'], test_data[1]), \
            "Test labels don't match!"

    print("✅ Verification passed! Data is identical.")
    return True


def create_backup(filepath: str) -> str:
    """
    Create a backup of the original file.

    Parameters:
    -----------
    filepath : str
        Path to the file to backup

    Returns:
    --------
    str
        Path to the backup file
    """
    backup_path = filepath + '.backup'
    print(f"\n💼 Creating backup: {backup_path}")
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backup created")
    return backup_path


def main():
    """Main conversion function."""
    print("=" * 60)
    print("MNIST Data Format Converter")
    print("Legacy pickle → Modern NPZ format")
    print("=" * 60)

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')

    legacy_path = os.path.join(data_dir, 'mnist.pkl.gz')
    npz_path = os.path.join(data_dir, 'mnist.npz')

    # Check if legacy file exists
    if not os.path.exists(legacy_path):
        print(f"❌ Error: Legacy file not found: {legacy_path}")
        sys.exit(1)

    # Check if NPZ already exists
    if os.path.exists(npz_path):
        response = input(f"\n⚠️  {npz_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Conversion cancelled.")
            sys.exit(0)

    try:
        # Step 1: Load legacy data
        original_data = load_legacy_mnist(legacy_path)

        # Step 2: Save as NPZ
        save_as_npz(original_data, npz_path)

        # Step 3: Verify conversion
        verify_conversion(npz_path, original_data)

        # Step 4: Create backup of original
        backup_path = create_backup(legacy_path)

        print("\n" + "=" * 60)
        print("✅ CONVERSION COMPLETE!")
        print("=" * 60)
        print(f"\n📁 Files:")
        print(f"   - New NPZ file: {npz_path}")
        print(f"   - Original backup: {backup_path}")
        print(f"\n📝 Next steps:")
        print(f"   1. Update src/mnist_loader.py to use the NPZ format")
        print(f"   2. Run tests: pytest")
        print(f"   3. If all tests pass, you can remove the backup")
        print(f"\n💡 To rollback: mv {backup_path} {legacy_path}")

    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
