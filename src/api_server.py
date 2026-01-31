"""
api_server.py
~~~~~~~~~~~~~

Flask-based REST API server with WebSocket support for neural network training.

This module provides endpoints for:
- Creating and managing neural networks
- Training networks with real-time progress updates via WebSockets
- Testing networks with MNIST digit recognition
- Persisting networks to/from SQLite database

The server uses:
- Flask for REST API endpoints
- Flask-SocketIO for WebSocket communication
- Gevent for async background training tasks
- SQLite for network persistence
"""

import os
import sys
import uuid
import base64
import logging
from io import BytesIO
from typing import Dict, Any, Tuple

import gevent
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

# Set matplotlib to use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use Agg backend which doesn't require a display
import matplotlib.pyplot as plt

# Import our existing network code from the src package
from src import network
from src import mnist_loader
from src.model_persistence import (
    save_network,
    list_saved_networks,
    delete_network,
    delete_old_networks
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize SocketIO for WebSocket support
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    logger=True,
    engineio_logger=True,
    ping_timeout=60,   # Seconds before considering connection dead
    ping_interval=25   # Send ping every 25 seconds to keep connection alive
)

# Global state storage
# Store active networks in memory: {network_id: {network, architecture, ...}}
active_networks: Dict[str, Dict[str, Any]] = {}

# Store training jobs: {job_id: {network_id, status, progress, ...}}
training_jobs: Dict[str, Dict[str, Any]] = {}

# MNIST dataset storage (loaded once at startup)
training_data = None
validation_data = None
test_data = None


def load_mnist_data() -> None:
    """
    Load MNIST dataset into global variables.

    This function loads the dataset once at startup to avoid repeated
    loading for each training operation.

    Raises:
        Exception: If data loading fails
    """
    global training_data, validation_data, test_data

    logger.info("Loading MNIST data...")
    try:
        training_data, validation_data, test_data = (
            mnist_loader.load_data_wrapper()
        )
        logger.info(
            f"Data loaded successfully! "
            f"Training: {len(training_data)}, "
            f"Validation: {len(validation_data)}, "
            f"Test: {len(test_data)}"
        )
    except Exception as e:
        logger.exception(f"Error loading MNIST data: {e}")
        raise


# Load data at module initialization
load_mnist_data()


def cleanup_old_networks_task() -> None:
    """
    Background task to periodically clean up old networks.

    This task runs continuously in the background and deletes networks
    older than 2 days every 24 hours. It helps prevent database bloat
    and removes stale test/training networks.
    """
    while True:
        try:
            # Wait 24 hours between cleanup runs
            # Using 86400 seconds (24 hours) as the interval
            gevent.sleep(86400)

            logger.info("Starting automatic cleanup of old networks...")
            deleted_count = delete_old_networks(days=2)

            if deleted_count > 0:
                logger.info(
                    f"Automatic cleanup completed: "
                    f"deleted {deleted_count} network(s)"
                )
            else:
                logger.debug("Automatic cleanup completed: no old networks found")

        except Exception as e:
            logger.exception(f"Error during automatic network cleanup: {e}")
            # Continue running despite errors
            continue


def start_cleanup_task() -> None:
    """
    Start the background cleanup task using gevent.

    This should be called once when the server starts.
    """
    logger.info("Starting automatic network cleanup task (runs every 24 hours)")
    socketio.start_background_task(cleanup_old_networks_task)


@app.route('/api/status', methods=['GET'])
def get_status() -> Tuple[Dict[str, Any], int]:
    """
    Get server status and statistics.

    Returns:
        tuple: JSON response with status information and HTTP status code

    Response:
        {
            'status': 'online',
            'active_networks': int,
            'training_jobs': int
        }
    """
    return jsonify({
        'status': 'online',
        'active_networks': len(active_networks),
        'training_jobs': len(training_jobs)
    }), 200

@app.route('/api/networks', methods=['POST'])
def create_network() -> Tuple[Dict[str, Any], int]:
    """
    Create a new neural network with the specified architecture.

    Request Body:
        {
            'layer_sizes': [784, 30, 10]  # Optional, defaults to [784, 30, 10]
        }

    Returns:
        tuple: JSON response with network details and HTTP status code

    Response:
        {
            'network_id': str,
            'architecture': list,
            'status': 'created'
        }
    """
    data = request.get_json() or {}

    # Get layer sizes from request, default to [784, 30, 10] if not specified
    layer_sizes = data.get('layer_sizes', [784, 30, 10])
    
    # Validate architecture
    if not isinstance(layer_sizes, list) or len(layer_sizes) < 2:
        logger.warning(f"Invalid architecture requested: {layer_sizes}")
        return jsonify({
            'error': 'Invalid architecture. Must have at least 2 layers.'
        }), 400

    # Create a unique ID for this network
    network_id = str(uuid.uuid4())
    
    try:
        # Create the network with specified architecture
        net = network.Network(layer_sizes)

        # Store in our dictionary
        active_networks[network_id] = {
            'network': net,
            'architecture': layer_sizes,
            'trained': False,
            'accuracy': None
        }

        logger.info(
            f"Created network {network_id} with architecture {layer_sizes}"
        )

        return jsonify({
            'network_id': network_id,
            'architecture': layer_sizes,
            'status': 'created'
        }), 201

    except Exception as e:
        logger.exception(f"Error creating network: {e}")
        return jsonify({'error': f'Failed to create network: {str(e)}'}), 500

@app.route('/api/networks/<network_id>/train', methods=['POST'])
def train_network(network_id: str) -> Tuple[Dict[str, Any], int]:
    """
    Start asynchronous training for the specified network.

    Args:
        network_id: UUID of the network to train

    Request Body:
        {
            'epochs': 5,              # Optional, default: 5
            'mini_batch_size': 10,    # Optional, default: 10
            'learning_rate': 3.0      # Optional, default: 3.0
        }

    Returns:
        tuple: JSON response with job details and HTTP status code

    Response:
        {
            'job_id': str,
            'network_id': str,
            'status': 'training_started'
        }
    """
    if network_id not in active_networks:
        logger.warning(f"Training requested for non-existent network: {network_id}")
        return jsonify({'error': 'Network not found'}), 404
        
    data = request.get_json() or {}
    epochs = data.get('epochs', 5)
    mini_batch_size = data.get('mini_batch_size', 10)
    learning_rate = data.get('learning_rate', 3.0)
    
    # Validate parameters
    if not isinstance(epochs, int) or epochs < 1:
        return jsonify({'error': 'epochs must be a positive integer'}), 400
    if not isinstance(mini_batch_size, int) or mini_batch_size < 1:
        return jsonify({'error': 'mini_batch_size must be a positive integer'}), 400
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        return jsonify({'error': 'learning_rate must be a positive number'}), 400

    # Create a job ID for this training task
    job_id = str(uuid.uuid4())
    
    # Set up the training job status
    training_jobs[job_id] = {
        'network_id': network_id,
        'status': 'pending',
        'progress': 0,
        'epochs': epochs
    }
    
    logger.info(
        f"Starting training job {job_id} for network {network_id}: "
        f"epochs={epochs}, batch_size={mini_batch_size}, "
        f"learning_rate={learning_rate}"
    )

    # Start training in a background task (using gevent)
    socketio.start_background_task(
        train_network_task,
        network_id, job_id, epochs, mini_batch_size, learning_rate
    )

    return jsonify({
        'job_id': job_id,
        'network_id': network_id,
        'status': 'training_started'
    }), 202

def train_network_task(
    network_id: str,
    job_id: str,
    epochs: int,
    mini_batch_size: int,
    learning_rate: float
) -> None:
    """
    Background task to train a neural network.

    This function runs in a separate gevent greenlet and sends progress
    updates via WebSocket as training progresses.

    Args:
        network_id: UUID of the network to train
        job_id: UUID of the training job
        epochs: Number of training epochs
        mini_batch_size: Size of mini-batches for SGD
        learning_rate: Learning rate for gradient descent
    """
    net = active_networks[network_id]['network']
    
    def epoch_callback(data: Dict[str, Any]) -> None:
        """
        Callback function for each epoch to send updates via WebSocket.

        Args:
            data: Dictionary containing epoch information
        """
        # Update the job status
        training_jobs[job_id]['status'] = 'training'
        training_jobs[job_id]['progress'] = (
            (data['epoch'] / data['total_epochs']) * 100
        )

        # Prepare update data for WebSocket emission
        update_data = {
            'job_id': job_id,
            'network_id': network_id,
            'epoch': data['epoch'],
            'total_epochs': data['total_epochs'],
            'accuracy': data['accuracy'],
            'elapsed_time': data['elapsed_time'],
            'progress': training_jobs[job_id]['progress'],
            'correct': data.get('correct'),
            'total': data.get('total')
        }
        
        logger.debug(
            f"Training progress - Job {job_id}: "
            f"Epoch {data['epoch']}/{data['total_epochs']}, "
            f"Accuracy: {data['accuracy']:.2%}"
        )

        # Emit the progress update through WebSocket
        socketio.emit('training_update', update_data)
        # Yield control to gevent to send the message immediately
        gevent.sleep(0)

    try:
        logger.info(f"Starting training for job {job_id}")

        # Train the network with the callback function
        net.SGD(
            training_data,
            epochs,
            mini_batch_size,
            learning_rate,
            test_data=test_data,
            callback=epoch_callback
        )

        # Calculate final accuracy
        accuracy = net.evaluate(test_data) / len(test_data)
        
        # Update network status
        active_networks[network_id]['trained'] = True
        active_networks[network_id]['accuracy'] = accuracy
        
        # Update job status
        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['accuracy'] = accuracy
        training_jobs[job_id]['progress'] = 100

        # Save the trained network with metadata
        save_network(net, network_id, trained=True, accuracy=accuracy)

        logger.info(
            f"Training completed for job {job_id}: "
            f"Final accuracy: {accuracy:.2%}"
        )

        # Emit completion event via WebSocket
        socketio.emit('training_complete', {
            'job_id': job_id,
            'network_id': network_id,
            'status': 'completed',
            'accuracy': float(accuracy),
            'progress': 100
        })
        # Yield control to gevent to send the message immediately
        gevent.sleep(0)

    except Exception as e:
        logger.exception(f"Training failed for job {job_id}: {e}")

        # Update job status on error
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = str(e)

        # Emit error event via WebSocket
        socketio.emit('training_error', {
            'job_id': job_id,
            'network_id': network_id,
            'status': 'failed',
            'error': str(e)
        })
        # Yield control to gevent to send the message immediately
        gevent.sleep(0)

@app.route('/api/training/<job_id>', methods=['GET'])
def get_training_status(job_id: str) -> Tuple[Dict[str, Any], int]:
    """
    Get the status of a training job.

    Args:
        job_id: UUID of the training job

    Returns:
        tuple: JSON response with job status and HTTP status code

    Response:
        {
            'network_id': str,
            'status': 'pending' | 'training' | 'completed' | 'failed',
            'progress': float,
            'epochs': int,
            'accuracy': float (if completed),
            'error': str (if failed)
        }
    """
    if job_id not in training_jobs:
        logger.warning(f"Status requested for non-existent job: {job_id}")
        return jsonify({'error': 'Training job not found'}), 404
    
    return jsonify(training_jobs[job_id]), 200

@app.route('/api/networks', methods=['GET'])
def list_networks() -> Tuple[Dict[str, Any], int]:
    """
    List all available networks (both in-memory and saved).

    Returns:
        tuple: JSON response with network list and HTTP status code

    Response:
        {
            'networks': [
                {
                    'network_id': str,
                    'architecture': list,
                    'trained': bool,
                    'accuracy': float,
                    'status': 'in_memory' | 'saved'
                },
                ...
            ]
        }
    """
    # Combine in-memory and saved networks
    in_memory = [
        {
            'network_id': nid,
            'architecture': info['architecture'],
            'trained': info['trained'],
            'accuracy': info['accuracy'],
            'status': 'in_memory'
        }
        for nid, info in active_networks.items()
    ]
    
    # Get saved networks, exclude ones already in memory to avoid duplicates
    in_memory_ids = set(active_networks.keys())
    saved = list_saved_networks()
    saved_not_in_memory = []
    for net in saved:
        if net['network_id'] not in in_memory_ids:
            net['status'] = 'saved'
            saved_not_in_memory.append(net)

    logger.debug(
        f"Listing networks: {len(in_memory)} in memory, "
        f"{len(saved_not_in_memory)} saved only"
    )

    return jsonify({
        'networks': in_memory + saved_not_in_memory
    }), 200

@app.route('/api/networks/<network_id>', methods=['DELETE'])
def delete_network_endpoint(network_id: str) -> Tuple[Dict[str, Any], int]:
    """
    Delete a network from both memory and disk.

    Args:
        network_id: UUID of the network to delete

    Returns:
        tuple: JSON response with deletion details and HTTP status code

    Response:
        {
            'network_id': str,
            'deleted_from_memory': bool,
            'deleted_from_disk': bool
        }
    """
    # Remove from active networks if present
    deleted_from_memory = False
    if network_id in active_networks:
        del active_networks[network_id]
        deleted_from_memory = True
    
    # Delete from disk if present
    deleted_from_disk = delete_network(network_id)
    
    if not deleted_from_memory and not deleted_from_disk:
        logger.warning(f"Delete attempted for non-existent network: {network_id}")
        return jsonify({'error': 'Network not found'}), 404
    
    logger.info(
        f"Deleted network {network_id}: "
        f"memory={deleted_from_memory}, disk={deleted_from_disk}"
    )

    return jsonify({
        'network_id': network_id,
        'deleted_from_memory': deleted_from_memory,
        'deleted_from_disk': deleted_from_disk
    }), 200

@app.route('/api/networks', methods=['DELETE'])
def delete_all_networks() -> Tuple[Dict[str, Any], int]:
    """
    Delete all networks from both memory and disk.

    Returns:
        tuple: JSON response with deletion summary and HTTP status code

    Response:
        {
            'deleted_count': int,
            'deleted_from_memory': int,
            'deleted_from_disk': int,
            'message': str
        }
    """
    # Get all network IDs (in-memory and saved)
    in_memory_ids = list(active_networks.keys())
    saved_networks = list_saved_networks()
    saved_ids = [net['network_id'] for net in saved_networks]

    # Combine and deduplicate
    all_network_ids = list(set(in_memory_ids + saved_ids))

    deleted_count = 0
    deleted_from_memory_count = 0
    deleted_from_disk_count = 0

    # Delete each network
    for network_id in all_network_ids:
        # Remove from active networks if present
        if network_id in active_networks:
            del active_networks[network_id]
            deleted_from_memory_count += 1

        # Delete from disk if present
        if delete_network(network_id):
            deleted_from_disk_count += 1

        deleted_count += 1

    logger.info(
        f"Deleted all networks: {deleted_count} total, "
        f"{deleted_from_memory_count} from memory, "
        f"{deleted_from_disk_count} from disk"
    )

    return jsonify({
        'deleted_count': deleted_count,
        'deleted_from_memory': deleted_from_memory_count,
        'deleted_from_disk': deleted_from_disk_count,
        'message': f'Successfully deleted {deleted_count} network(s)'
    }), 200


@app.route('/api/networks/cleanup', methods=['POST'])
def cleanup_old_networks_endpoint() -> Tuple[Dict[str, Any], int]:
    """
    Manually trigger cleanup of networks older than specified days.

    Request Body:
        {
            'days': 2  # Optional, default: 2
        }

    Returns:
        tuple: JSON response with cleanup summary and HTTP status code

    Response:
        {
            'deleted_count': int,
            'days': int,
            'message': str
        }
    """
    data = request.get_json() or {}
    days = data.get('days', 2)

    # Validate days parameter
    if not isinstance(days, (int, float)) or days < 0:
        return jsonify({
            'error': 'days must be a non-negative number'
        }), 400

    try:
        deleted_count = delete_old_networks(days=int(days))

        if deleted_count == -1:
            return jsonify({
                'error': 'Error occurred during cleanup'
            }), 500

        logger.info(
            f"Manual cleanup triggered: deleted {deleted_count} network(s) "
            f"older than {days} day(s)"
        )

        return jsonify({
            'deleted_count': deleted_count,
            'days': days,
            'message': (
                f'Successfully deleted {deleted_count} network(s) '
                f'older than {days} day(s)'
            )
        }), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"Error during manual cleanup: {e}")
        return jsonify({'error': 'Internal server error'}), 500


def _convert_to_float_list(array: np.ndarray) -> list:
    """
    Convert numpy array to list of floats for JSON serialization.

    Args:
        array: Numpy array to convert

    Returns:
        list: List of float values
    """
    return [
        float(val.item()) if hasattr(val, 'item') else float(val)
        for val in array.flatten()
    ]


def _create_digit_image(
    image_data: np.ndarray,
    predicted: int,
    actual: int
) -> str:
    """
    Create a base64-encoded PNG image of a digit.

    Args:
        image_data: 784-element array representing the digit image
        predicted: Predicted digit (0-9)
        actual: Actual digit (0-9)

    Returns:
        str: Base64-encoded PNG image
    """
    plt.figure(figsize=(3, 3))
    plt.imshow(image_data.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted} | Actual: {actual}")
    plt.axis('off')

    # Save image to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return img_base64


@app.route('/api/networks/<network_id>/successful_example', methods=['GET'])
def get_successful_example(
    network_id: str
) -> Tuple[Dict[str, Any], int]:
    """
    Return a random successful example prediction with network output details.

    Args:
        network_id: UUID of the network

    Returns:
        tuple: JSON response with example details and HTTP status code

    Response:
        {
            'network_id': str,
            'example_index': int,
            'predicted_digit': int,
            'actual_digit': int,
            'image_data': str (base64),
            'output_weights': list,
            'network_output': list
        }
    """
    if network_id not in active_networks:
        logger.warning(
            f"Successful example requested for non-existent network: "
            f"{network_id}"
        )
        return jsonify({'error': 'Network not found'}), 404
        
    net = active_networks[network_id]['network']
    
    # Find a successful example
    max_attempts = 100
    for attempt in range(max_attempts):
        # Choose a random example from test data
        index = np.random.randint(0, len(test_data))
        x, y = test_data[index]
        
        # Get the network's output
        output = net.feedforward(x)
        predicted_digit = int(np.argmax(output))
        actual_digit = int(y)
        
        if predicted_digit == actual_digit:
            logger.debug(
                f"Found successful example on attempt {attempt + 1}"
            )

            # Create image
            img_base64 = _create_digit_image(x, predicted_digit, actual_digit)

            # Get output layer weights
            output_weights = net.weights[-1].tolist()

            return jsonify({
                'network_id': network_id,
                'example_index': index,
                'predicted_digit': predicted_digit,
                'actual_digit': actual_digit,
                'image_data': img_base64,
                'output_weights': output_weights,
                'network_output': _convert_to_float_list(output)
            }), 200

    logger.warning(
        f"No successful example found after {max_attempts} attempts "
        f"for network {network_id}"
    )
    return jsonify({
        'error': f'No successful example found after {max_attempts} attempts'
    }), 404

@app.route('/api/networks/<network_id>/unsuccessful_example', methods=['GET'])
def get_unsuccessful_example(
    network_id: str
) -> Tuple[Dict[str, Any], int]:
    """
    Return a random unsuccessful example prediction with network details.

    Args:
        network_id: UUID of the network

    Returns:
        tuple: JSON response with example details and HTTP status code

    Response:
        {
            'network_id': str,
            'example_index': int,
            'predicted_digit': int,
            'actual_digit': int,
            'image_data': str (base64),
            'output_weights': list,
            'network_output': list
        }
    """
    if network_id not in active_networks:
        logger.warning(
            f"Unsuccessful example requested for non-existent network: "
            f"{network_id}"
        )
        return jsonify({'error': 'Network not found'}), 404
        
    net = active_networks[network_id]['network']
    
    # Find an unsuccessful example
    max_attempts = 200
    for attempt in range(max_attempts):
        # Choose a random example from test data
        index = np.random.randint(0, len(test_data))
        x, y = test_data[index]
        
        # Get the network's output
        output = net.feedforward(x)
        predicted_digit = int(np.argmax(output))
        actual_digit = int(y)
        
        if predicted_digit != actual_digit:
            logger.debug(
                f"Found unsuccessful example on attempt {attempt + 1}"
            )

            # Create image
            img_base64 = _create_digit_image(x, predicted_digit, actual_digit)

            # Get output layer weights
            output_weights = net.weights[-1].tolist()

            return jsonify({
                'network_id': network_id,
                'example_index': index,
                'predicted_digit': predicted_digit,
                'actual_digit': actual_digit,
                'image_data': img_base64,
                'output_weights': output_weights,
                'network_output': _convert_to_float_list(output)
            }), 200

    logger.warning(
        f"No unsuccessful example found after {max_attempts} attempts "
        f"for network {network_id}"
    )
    return jsonify({
        'error': f'No unsuccessful example found after {max_attempts} attempts'
    }), 404

@app.route('/')
def index() -> Any:
    """
    Serve the main frontend page.

    Returns:
        HTML response with the index page
    """
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path: str) -> Any:
    """
    Serve static files.

    Args:
        path: Path to the static file

    Returns:
        Static file response
    """
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    # Make sure the static directory exists
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        logger.info(f"Created static directory: {static_dir}")

    # Check if we're running in a cloud environment
    is_production = bool(
        os.environ.get('RAILWAY_STATIC_URL') or
        os.environ.get('PORT')
    )

    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8000))
    
    # Log server startup information
    if is_production:
        logger.info(f"Starting server in production mode on port {port}")
    else:
        logger.info(f"Starting server at http://localhost:{port}/")

    # Start the background cleanup task
    start_cleanup_task()

    # Use SocketIO for running the app (enables WebSocket support)
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            debug=not is_production,
            use_reloader=False,
            allow_unsafe_werkzeug=True
        )
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(
                f"Port {port} is already in use. "
                f"Please terminate the other process first."
            )
            logger.info(
                "You can use: pkill -f 'python src/api_server.py'"
            )
            sys.exit(1)
        else:
            raise
