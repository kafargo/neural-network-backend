"""
test_model_persistence.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for SQLite-based model persistence.
"""

import pytest
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network import Network
from model_persistence import (
    save_network,
    load_network,
    list_saved_networks,
    delete_network,
    get_network_metadata,
    delete_old_networks,
    ModelDatabase
)


@pytest.fixture
def temp_db_dir(tmp_path):
    """Create a temporary directory for database storage."""
    db_dir = tmp_path / "test_models"
    db_dir.mkdir()
    return str(db_dir)


@pytest.fixture
def simple_network():
    """Create a simple 3-layer network for testing."""
    return Network([3, 4, 2])


@pytest.fixture
def trained_network(simple_network):
    """Create a simple network with some training applied."""
    import numpy as np

    # Create minimal training data
    training_data = []
    for i in range(10):
        x = np.random.randn(3, 1)
        y = np.zeros((2, 1))
        y[i % 2] = 1.0
        training_data.append((x, y))

    # Train briefly
    simple_network.SGD(training_data, epochs=1, mini_batch_size=5, eta=0.1)
    return simple_network


@pytest.mark.unit
class TestModelPersistence:
    """Test basic model persistence operations."""

    def test_save_network_creates_database(self, simple_network, temp_db_dir):
        """Test that saving a network creates the database file."""
        network_id = "test_network_1"

        success = save_network(
            simple_network,
            network_id,
            model_dir=temp_db_dir,
            trained=False
        )

        assert success is True
        assert os.path.exists(f"{temp_db_dir}/networks.db")

    def test_save_network_with_metadata(self, trained_network, temp_db_dir):
        """Test that network metadata is saved correctly."""
        network_id = "trained_network_1"
        accuracy = 0.85

        success = save_network(
            trained_network,
            network_id,
            model_dir=temp_db_dir,
            trained=True,
            accuracy=accuracy
        )

        assert success is True

        # Verify metadata
        metadata = get_network_metadata(network_id, temp_db_dir)
        assert metadata is not None
        assert metadata['network_id'] == network_id
        assert metadata['trained'] is True
        assert metadata['accuracy'] == accuracy
        assert metadata['architecture'] == [3, 4, 2]

    def test_load_network_returns_network(self, simple_network, temp_db_dir):
        """Test that loading a network returns a valid Network object."""
        network_id = "test_network_2"

        save_network(simple_network, network_id, model_dir=temp_db_dir)
        loaded_network = load_network(network_id, temp_db_dir)

        assert loaded_network is not None
        assert isinstance(loaded_network, Network)
        assert loaded_network.sizes == simple_network.sizes

    def test_load_nonexistent_network(self, temp_db_dir):
        """Test that loading a non-existent network returns None."""
        loaded_network = load_network("nonexistent", temp_db_dir)
        assert loaded_network is None

    def test_load_preserves_weights(self, trained_network, temp_db_dir):
        """Test that saved weights are preserved after loading."""
        network_id = "test_network_3"

        save_network(trained_network, network_id, model_dir=temp_db_dir)
        loaded_network = load_network(network_id, temp_db_dir)

        # Compare weights
        import numpy as np
        for original_w, loaded_w in zip(trained_network.weights, loaded_network.weights):
            assert np.allclose(original_w, loaded_w)

        # Compare biases
        for original_b, loaded_b in zip(trained_network.biases, loaded_network.biases):
            assert np.allclose(original_b, loaded_b)

    def test_list_saved_networks_empty(self, temp_db_dir):
        """Test listing networks when database is empty."""
        networks = list_saved_networks(temp_db_dir)
        assert networks == []

    def test_list_saved_networks(self, simple_network, temp_db_dir):
        """Test that listing networks returns correct metadata."""
        # Save multiple networks
        save_network(simple_network, "net1", model_dir=temp_db_dir, trained=True, accuracy=0.9)
        save_network(simple_network, "net2", model_dir=temp_db_dir, trained=False)

        networks = list_saved_networks(temp_db_dir)

        assert len(networks) == 2
        assert any(net['network_id'] == "net1" for net in networks)
        assert any(net['network_id'] == "net2" for net in networks)

    def test_list_saved_networks_includes_metadata(self, simple_network, temp_db_dir):
        """Test that listed networks include all expected metadata fields."""
        network_id = "metadata_test"
        accuracy = 0.75

        save_network(
            simple_network,
            network_id,
            model_dir=temp_db_dir,
            trained=True,
            accuracy=accuracy
        )

        networks = list_saved_networks(temp_db_dir)
        network = networks[0]

        assert network['network_id'] == network_id
        assert network['architecture'] == [3, 4, 2]
        assert network['trained'] is True
        assert network['accuracy'] == accuracy
        assert 'created_at' in network
        assert 'updated_at' in network
        assert 'weights_shape' in network
        assert 'biases_shape' in network

    def test_delete_network_success(self, simple_network, temp_db_dir):
        """Test successful network deletion."""
        network_id = "delete_test"

        save_network(simple_network, network_id, model_dir=temp_db_dir)

        # Verify it exists
        assert load_network(network_id, temp_db_dir) is not None

        # Delete it
        success = delete_network(network_id, temp_db_dir)
        assert success is True

        # Verify it's gone
        assert load_network(network_id, temp_db_dir) is None

    def test_delete_nonexistent_network(self, temp_db_dir):
        """Test that deleting a non-existent network returns False."""
        # Initialize database
        db = ModelDatabase(db_path=f'{temp_db_dir}/networks.db')

        success = delete_network("nonexistent", temp_db_dir)
        assert success is False

    def test_save_untrained_network(self, simple_network, temp_db_dir):
        """Test saving a network that hasn't been trained."""
        network_id = "untrained_test"

        success = save_network(
            simple_network,
            network_id,
            model_dir=temp_db_dir,
            trained=False,
            accuracy=None
        )

        assert success is True

        metadata = get_network_metadata(network_id, temp_db_dir)
        assert metadata['trained'] is False
        assert metadata['accuracy'] is None

    def test_update_network(self, simple_network, temp_db_dir):
        """Test that saving a network with the same ID updates it."""
        network_id = "update_test"

        # Save untrained
        save_network(
            simple_network,
            network_id,
            model_dir=temp_db_dir,
            trained=False
        )

        metadata1 = get_network_metadata(network_id, temp_db_dir)
        assert metadata1['trained'] is False

        # Update to trained
        save_network(
            simple_network,
            network_id,
            model_dir=temp_db_dir,
            trained=True,
            accuracy=0.88
        )

        metadata2 = get_network_metadata(network_id, temp_db_dir)
        assert metadata2['trained'] is True
        assert metadata2['accuracy'] == 0.88

        # Should still be only one network
        networks = list_saved_networks(temp_db_dir)
        assert len(networks) == 1


@pytest.mark.integration
class TestPersistenceIntegration:
    """Integration tests for model persistence."""

    def test_save_load_train_cycle(self, simple_network, temp_db_dir):
        """Test complete cycle: save, load, train, save again."""
        import numpy as np

        network_id = "cycle_test"

        # Save initial untrained network
        save_network(
            simple_network,
            network_id,
            model_dir=temp_db_dir,
            trained=False
        )

        # Load it
        loaded_network = load_network(network_id, temp_db_dir)

        # Train it
        training_data = []
        for i in range(10):
            x = np.random.randn(3, 1)
            y = np.zeros((2, 1))
            y[i % 2] = 1.0
            training_data.append((x, y))

        loaded_network.SGD(training_data, epochs=1, mini_batch_size=5, eta=0.1)

        # Save trained version
        save_network(
            loaded_network,
            network_id,
            model_dir=temp_db_dir,
            trained=True,
            accuracy=0.85
        )

        # Load again and verify
        final_network = load_network(network_id, temp_db_dir)
        metadata = get_network_metadata(network_id, temp_db_dir)

        assert final_network is not None
        assert metadata['trained'] is True
        assert metadata['accuracy'] == 0.85

    def test_multiple_networks_coexist(self, temp_db_dir):
        """Test that multiple networks can coexist in the database."""
        networks_to_create = [
            ([784, 30, 10], "mnist_network"),
            ([3, 4, 2], "simple_network"),
            ([10, 20, 20, 10], "deep_network")
        ]

        # Create and save multiple networks
        for architecture, network_id in networks_to_create:
            net = Network(architecture)
            save_network(net, network_id, model_dir=temp_db_dir)

        # List all networks
        networks = list_saved_networks(temp_db_dir)
        assert len(networks) == len(networks_to_create)

        # Verify each can be loaded
        for architecture, network_id in networks_to_create:
            loaded = load_network(network_id, temp_db_dir)
            assert loaded is not None
            assert loaded.sizes == architecture

    def test_concurrent_access_safe(self, simple_network, temp_db_dir):
        """Test that database handles multiple operations safely."""
        # This tests that the context manager properly handles transactions
        network_ids = [f"concurrent_{i}" for i in range(5)]

        # Save multiple networks
        for network_id in network_ids:
            save_network(simple_network, network_id, model_dir=temp_db_dir)

        # Load all of them
        loaded_networks = []
        for network_id in network_ids:
            net = load_network(network_id, temp_db_dir)
            loaded_networks.append(net)

        assert len(loaded_networks) == 5
        assert all(net is not None for net in loaded_networks)

        # Delete them all
        for network_id in network_ids:
            assert delete_network(network_id, temp_db_dir) is True

        # Verify all deleted
        networks = list_saved_networks(temp_db_dir)
        assert len(networks) == 0


class TestDeleteOldNetworks:
    """Tests for automatic cleanup of old networks."""

    def test_delete_old_networks_basic(self, simple_network, temp_db_dir):
        """Test basic delete_old_networks functionality."""
        import time
        import sqlite3

        # Save a network
        network_id = "test_network"
        save_network(simple_network, network_id, model_dir=temp_db_dir)

        # Manually update the created_at timestamp to 3 days ago
        db_path = os.path.join(temp_db_dir, "networks.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE networks 
            SET created_at = datetime('now', '-3 days')
            WHERE network_id = ?
        ''', (network_id,))
        conn.commit()
        conn.close()

        # Delete networks older than 2 days
        deleted_count = delete_old_networks(days=2, model_dir=temp_db_dir)

        assert deleted_count == 1
        assert load_network(network_id, temp_db_dir) is None

    def test_delete_old_networks_preserves_recent(
        self,
        simple_network,
        temp_db_dir
    ):
        """Test that recent networks are not deleted."""
        # Save a new network
        network_id = "recent_network"
        save_network(simple_network, network_id, model_dir=temp_db_dir)

        # Try to delete old networks
        deleted_count = delete_old_networks(days=2, model_dir=temp_db_dir)

        # Should not delete recent network
        assert deleted_count == 0
        assert load_network(network_id, temp_db_dir) is not None

    def test_delete_old_networks_mixed_ages(
        self,
        simple_network,
        temp_db_dir
    ):
        """Test with a mix of old and recent networks."""
        import sqlite3

        # Create multiple networks
        old_ids = ["old_1", "old_2"]
        recent_ids = ["recent_1", "recent_2"]

        for network_id in old_ids + recent_ids:
            save_network(simple_network, network_id, model_dir=temp_db_dir)

        # Manually age the old networks
        db_path = os.path.join(temp_db_dir, "networks.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for network_id in old_ids:
            cursor.execute('''
                UPDATE networks 
                SET created_at = datetime('now', '-3 days')
                WHERE network_id = ?
            ''', (network_id,))
        conn.commit()
        conn.close()

        # Delete networks older than 2 days
        deleted_count = delete_old_networks(days=2, model_dir=temp_db_dir)

        # Verify old ones deleted, recent ones remain
        assert deleted_count == len(old_ids)
        for network_id in old_ids:
            assert load_network(network_id, temp_db_dir) is None
        for network_id in recent_ids:
            assert load_network(network_id, temp_db_dir) is not None

    def test_delete_old_networks_custom_days(
        self,
        simple_network,
        temp_db_dir
    ):
        """Test delete_old_networks with different day thresholds."""
        import sqlite3

        network_id = "test_network"
        save_network(simple_network, network_id, model_dir=temp_db_dir)

        # Age the network to 5 days old
        db_path = os.path.join(temp_db_dir, "networks.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE networks 
            SET created_at = datetime('now', '-5 days')
            WHERE network_id = ?
        ''', (network_id,))
        conn.commit()
        conn.close()

        # Should not delete (older than 7 days)
        deleted_count = delete_old_networks(days=7, model_dir=temp_db_dir)
        assert deleted_count == 0
        assert load_network(network_id, temp_db_dir) is not None

        # Should delete (older than 3 days)
        deleted_count = delete_old_networks(days=3, model_dir=temp_db_dir)
        assert deleted_count == 1
        assert load_network(network_id, temp_db_dir) is None

    def test_delete_old_networks_empty_db(self, temp_db_dir):
        """Test delete_old_networks on empty database."""
        deleted_count = delete_old_networks(days=2, model_dir=temp_db_dir)
        assert deleted_count == 0

    def test_delete_old_networks_negative_days(self, temp_db_dir):
        """Test that negative days raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            delete_old_networks(days=-1, model_dir=temp_db_dir)
        assert "non-negative" in str(exc_info.value)

    def test_delete_old_networks_zero_days(
        self,
        simple_network,
        temp_db_dir
    ):
        """Test delete_old_networks with days=0."""
        import sqlite3

        # Save a network
        network_id = "test_network"
        save_network(simple_network, network_id, model_dir=temp_db_dir)

        # Age it slightly (1 hour ago)
        db_path = os.path.join(temp_db_dir, "networks.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE networks 
            SET created_at = datetime('now', '-1 hour')
            WHERE network_id = ?
        ''', (network_id,))
        conn.commit()
        conn.close()

        # Should delete anything older than now
        deleted_count = delete_old_networks(days=0, model_dir=temp_db_dir)
        assert deleted_count == 1
        assert load_network(network_id, temp_db_dir) is None

    def test_model_database_delete_old_networks_method(self, temp_db_dir):
        """Test ModelDatabase.delete_old_networks_from_db directly."""
        import sqlite3

        db = ModelDatabase(db_path=os.path.join(temp_db_dir, "networks.db"))
        net = Network([3, 4, 2])

        # Save network
        db.save_network_to_db(net, "test_network", trained=False)

        # Age it
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE networks 
            SET created_at = datetime('now', '-3 days')
            WHERE network_id = ?
        ''', ("test_network",))
        conn.commit()
        conn.close()

        # Delete old networks
        deleted = db.delete_old_networks_from_db(days=2)
        assert deleted == 1
        assert db.load_network_from_db("test_network") is None

    def test_delete_old_networks_two_weeks_old(
        self,
        simple_network,
        temp_db_dir
    ):
        """
        Test delete_old_networks with networks that are 14 days old.

        This test simulates the production issue where networks created
        two weeks ago should be deleted when using the default 2-day threshold.
        """
        import sqlite3

        network_id = "two_weeks_old_network"
        save_network(simple_network, network_id, model_dir=temp_db_dir)

        # Age the network to 14 days old
        db_path = os.path.join(temp_db_dir, "networks.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE networks 
            SET created_at = datetime('now', '-14 days')
            WHERE network_id = ?
        ''', (network_id,))
        conn.commit()
        conn.close()

        # Verify the network age using the same query as delete_old_networks
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT network_id, created_at,
                   ROUND(julianday('now') - julianday(created_at), 2) as age_days
            FROM networks
            WHERE network_id = ?
        ''', (network_id,))
        row = cursor.fetchone()
        conn.close()

        # Network should be approximately 14 days old
        assert row['age_days'] >= 13.9, f"Network age should be ~14 days, got {row['age_days']}"

        # Delete networks older than 2 days (default threshold)
        deleted_count = delete_old_networks(days=2, model_dir=temp_db_dir)

        # The 14-day-old network should definitely be deleted
        assert deleted_count == 1
        assert load_network(network_id, temp_db_dir) is None

