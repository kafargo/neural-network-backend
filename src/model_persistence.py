"""
model_persistence.py
~~~~~~~~~~~~~~~~~~~~

SQLite-based persistence for neural network models.
Provides reliable, performant storage with ACID transaction guarantees.
"""

import sqlite3
import pickle
import json
import os
import logging
from typing import Optional, List, Dict, Any, Generator
from contextlib import contextmanager
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


class NetworkEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays."""

    def default(self, obj: Any) -> Any:
        """
        Convert numpy arrays to lists for JSON serialization.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable representation
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ModelDatabase:
    """
    Manages SQLite database for neural network model persistence.

    The database stores:
    - Network metadata (architecture, training status, accuracy)
    - Serialized network objects as binary blobs
    """

    def __init__(self, db_path: str = 'models/networks.db'):
        """
        Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_directory()
        self._initialize_schema()

    def _ensure_directory(self) -> None:
        """Create the database directory if it doesn't exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database connections.

        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        """Create the database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS networks (
                    network_id TEXT PRIMARY KEY,
                    architecture TEXT NOT NULL,
                    network_data BLOB NOT NULL,
                    trained INTEGER NOT NULL DEFAULT 0,
                    accuracy REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create index for common queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_trained 
                ON networks(trained)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON networks(created_at DESC)
            ''')

    def save_network_to_db(
        self,
        network,
        network_id: str,
        trained: bool = True,
        accuracy: Optional[float] = None
    ) -> bool:
        """
        Save a network to the database.

        Args:
            network: Network object to save
            network_id: Unique identifier for the network
            trained: Whether the network has been trained
            accuracy: Training accuracy (0.0 to 1.0)

        Returns:
            bool: True if successful, False otherwise

        Raises:
            ValueError: If accuracy is out of valid range
        """
        # Validate inputs
        if accuracy is not None and not 0.0 <= accuracy <= 1.0:
            raise ValueError(
                f"Accuracy must be between 0.0 and 1.0, got {accuracy}"
            )

        # Serialize the network object using pickle
        network_data = pickle.dumps(network)

        # Serialize architecture as JSON for queryability
        architecture_json = json.dumps(network.sizes, cls=NetworkEncoder)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO networks 
                (network_id, architecture, network_data, trained, 
                 accuracy, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                network_id,
                architecture_json,
                network_data,
                1 if trained else 0,
                accuracy
            ))

        logger.info(
            f"Saved network '{network_id}' with architecture "
            f"{network.sizes}, trained={trained}, accuracy={accuracy}"
        )
        return True

    def load_network_from_db(self, network_id: str):
        """
        Load a network from the database.

        Args:
            network_id: Unique identifier of the network

        Returns:
            Network object or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT network_data FROM networks WHERE network_id = ?',
                (network_id,)
            )
            row = cursor.fetchone()

            if row is None:
                logger.warning(f"Network '{network_id}' not found")
                return None

            # Deserialize the network object
            network = pickle.loads(row['network_data'])
            logger.info(f"Loaded network '{network_id}'")
            return network

    def list_networks_from_db(self) -> List[Dict[str, Any]]:
        """
        List all networks with metadata.

        Returns:
            List of network metadata dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    network_id,
                    architecture,
                    trained,
                    accuracy,
                    created_at,
                    updated_at
                FROM networks
                ORDER BY created_at DESC
            ''')

            networks = []
            for row in cursor.fetchall():
                architecture = json.loads(row['architecture'])

                # Calculate weight and bias shapes from architecture
                weights_shape = [
                    [architecture[i+1], architecture[i]]
                    for i in range(len(architecture) - 1)
                ]
                biases_shape = [
                    [architecture[i+1], 1]
                    for i in range(len(architecture) - 1)
                ]

                networks.append({
                    'network_id': row['network_id'],
                    'architecture': architecture,
                    'weights_shape': weights_shape,
                    'biases_shape': biases_shape,
                    'trained': bool(row['trained']),
                    'accuracy': row['accuracy'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                })

            logger.debug(f"Listed {len(networks)} networks")
            return networks

    def delete_network_from_db(self, network_id: str) -> bool:
        """
        Delete a network from the database.

        Args:
            network_id: Unique identifier of the network

        Returns:
            bool: True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM networks WHERE network_id = ?',
                (network_id,)
            )

            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted network '{network_id}'")
            else:
                logger.warning(
                    f"Could not delete network '{network_id}': not found"
                )
            return deleted

    def get_network_metadata_from_db(
        self,
        network_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get network metadata without loading the full object.

        Args:
            network_id: Unique identifier of the network

        Returns:
            Metadata dictionary or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    network_id,
                    architecture,
                    trained,
                    accuracy,
                    created_at,
                    updated_at
                FROM networks
                WHERE network_id = ?
            ''', (network_id,))

            row = cursor.fetchone()
            if row is None:
                logger.warning(
                    f"Metadata for network '{network_id}' not found"
                )
                return None

            architecture = json.loads(row['architecture'])

            return {
                'network_id': row['network_id'],
                'architecture': architecture,
                'trained': bool(row['trained']),
                'accuracy': row['accuracy'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            }


# Global database instance
_db = None


def _get_db() -> ModelDatabase:
    """
    Get or create the global database instance.

    Returns:
        ModelDatabase: The global database instance
    """
    global _db
    if _db is None:
        _db = ModelDatabase()
    return _db


def save_network(
    network,
    network_id: str,
    model_dir: str = 'models',
    trained: bool = True,
    accuracy: Optional[float] = None
) -> bool:
    """
    Save a neural network to the SQLite database.

    Args:
        network: The neural network object to save
        network_id: A unique identifier for the network
        model_dir: Directory for the database file
        trained: Boolean indicating if the network has been trained
        accuracy: The accuracy of the trained network (0.0 to 1.0)

    Returns:
        bool: True if the save was successful, False otherwise

    Raises:
        ValueError: If accuracy is out of valid range

    Example:
        >>> net = Network([784, 30, 10])
        >>> save_network(net, "my_network", trained=False)
        True
    """
    if not network_id or not isinstance(network_id, str):
        logger.error("Invalid network_id: must be a non-empty string")
        return False

    try:
        # Use singleton if default path, otherwise create new instance
        if model_dir == 'models':
            db = _get_db()
        else:
            db = ModelDatabase(db_path=f'{model_dir}/networks.db')

        return db.save_network_to_db(network, network_id, trained, accuracy)

    except ValueError as e:
        logger.error(f"Validation error saving network '{network_id}': {e}")
        return False
    except (AttributeError, pickle.PicklingError) as e:
        logger.error(
            f"Serialization error saving network '{network_id}': {e}"
        )
        return False
    except sqlite3.Error as e:
        logger.error(f"Database error saving network '{network_id}': {e}")
        return False
    except Exception as e:
        logger.exception(
            f"Unexpected error saving network '{network_id}': {e}"
        )
        return False


def load_network(network_id: str, model_dir: str = 'models'):
    """
    Load a neural network from the SQLite database.

    Args:
        network_id: The unique identifier of the network to load
        model_dir: Directory where the database is stored

    Returns:
        The loaded neural network object or None if not found

    Example:
        >>> net = load_network("my_network")
        >>> if net:
        ...     print(f"Loaded network with {len(net.sizes)} layers")
    """
    if not network_id or not isinstance(network_id, str):
        logger.error("Invalid network_id: must be a non-empty string")
        return None

    try:
        # Use singleton if default path, otherwise create new instance
        if model_dir == 'models':
            db = _get_db()
        else:
            db = ModelDatabase(db_path=f'{model_dir}/networks.db')

        return db.load_network_from_db(network_id)

    except pickle.UnpicklingError as e:
        logger.error(
            f"Deserialization error loading network '{network_id}': {e}"
        )
        return None
    except sqlite3.Error as e:
        logger.error(
            f"Database error loading network '{network_id}': {e}"
        )
        return None
    except Exception as e:
        logger.exception(
            f"Unexpected error loading network '{network_id}': {e}"
        )
        return None


def list_saved_networks(
    model_dir: str = 'models'
) -> List[Dict[str, Any]]:
    """
    List all saved networks with their metadata.

    Args:
        model_dir: Directory where the database is stored

    Returns:
        list: A list of metadata dictionaries for each saved network

    Example:
        >>> networks = list_saved_networks()
        >>> for net in networks:
        ...     print(f"{net['network_id']}: {net['architecture']}")
    """
    try:
        # Use singleton if default path, otherwise create new instance
        if model_dir == 'models':
            db = _get_db()
        else:
            db = ModelDatabase(db_path=f'{model_dir}/networks.db')

        return db.list_networks_from_db()

    except sqlite3.Error as e:
        logger.error(f"Database error listing networks: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error listing networks: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error listing networks: {e}")
        return []


def delete_network(network_id: str, model_dir: str = 'models') -> bool:
    """
    Delete a saved network from the database.

    Args:
        network_id: The unique identifier of the network to delete
        model_dir: Directory where the database is stored

    Returns:
        bool: True if deletion was successful, False otherwise

    Example:
        >>> if delete_network("old_network"):
        ...     print("Network deleted successfully")
    """
    if not network_id or not isinstance(network_id, str):
        logger.error("Invalid network_id: must be a non-empty string")
        return False

    try:
        # Use singleton if default path, otherwise create new instance
        if model_dir == 'models':
            db = _get_db()
        else:
            db = ModelDatabase(db_path=f'{model_dir}/networks.db')

        return db.delete_network_from_db(network_id)

    except sqlite3.Error as e:
        logger.error(f"Database error deleting network '{network_id}': {e}")
        return False
    except Exception as e:
        logger.exception(
            f"Unexpected error deleting network '{network_id}': {e}"
        )
        return False


def get_network_metadata(
    network_id: str,
    model_dir: str = 'models'
) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific network without loading full network object.

    Args:
        network_id: The unique identifier of the network
        model_dir: Directory where the database is stored

    Returns:
        dict: Network metadata or None if not found

    Example:
        >>> metadata = get_network_metadata("my_network")
        >>> if metadata:
        ...     print(f"Accuracy: {metadata['accuracy']}")
    """
    if not network_id or not isinstance(network_id, str):
        logger.error("Invalid network_id: must be a non-empty string")
        return None

    try:
        # Use singleton if default path, otherwise create new instance
        if model_dir == 'models':
            db = _get_db()
        else:
            db = ModelDatabase(db_path=f'{model_dir}/networks.db')

        return db.get_network_metadata_from_db(network_id)

    except sqlite3.Error as e:
        logger.error(
            f"Database error getting metadata for '{network_id}': {e}"
        )
        return None
    except json.JSONDecodeError as e:
        logger.error(
            f"JSON decode error getting metadata for '{network_id}': {e}"
        )
        return None
    except Exception as e:
        logger.exception(
            f"Unexpected error getting metadata for '{network_id}': {e}"
        )
        return None
