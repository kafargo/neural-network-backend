# Migration Assessment: Moving to Modern LTS Dependencies

**Date**: January 30, 2026  
**Status**: Assessment Complete

## Executive Summary

**Overall Difficulty**: ⭐⭐⭐ **MODERATE** (3/5)

The migration involves two main components:
1. **MNIST Data Format** - Easy (1-2 hours)
2. **Eventlet → Modern Async** - Moderate (4-8 hours)

**Total Estimated Effort**: 6-10 hours for a complete migration

---

## 1. MNIST Data Format Migration

### Current State
- Using legacy `mnist.pkl.gz` file created with NumPy 1.x
- Triggers `VisibleDeprecationWarning` in NumPy 2.4+
- File uses Python 2 pickle protocol with `latin1` encoding

### Migration Difficulty: ⭐ **EASY**

### Solution Options

#### Option A: Convert to NPZ Format (Recommended)
**Difficulty**: Very Easy  
**Time**: 30 minutes  
**Benefits**: 
- Native NumPy format
- Better compression
- Forward compatible
- No warnings

**Steps**:
1. Load existing pickle data
2. Save as `.npz` file using `np.savez_compressed()`
3. Update `mnist_loader.py` to use `np.load()`

**Code Change**:
```python
# New loader function
def load_data():
    data_path = os.path.join(base_path, 'data', 'mnist.npz')
    with np.load(data_path) as data:
        training_data = (data['train_images'], data['train_labels'])
        validation_data = (data['val_images'], data['val_labels'])
        test_data = (data['test_images'], data['test_labels'])
    return (training_data, validation_data, test_data)
```

#### Option B: Use torchvision/tensorflow-datasets
**Difficulty**: Easy-Moderate  
**Time**: 1-2 hours  
**Benefits**:
- Automatic downloads
- Always up-to-date
- Many other datasets available

**Drawbacks**:
- Adds heavy dependency (torch/tensorflow)
- Changes data format slightly

#### Option C: Keep Current + Regenerate Pickle
**Difficulty**: Very Easy  
**Time**: 15 minutes  
**Steps**:
1. Load existing data with warnings suppressed
2. Save with modern pickle protocol: `pickle.dump(data, f, protocol=4)`

---

## 2. Eventlet → Modern Async Migration

### Current State
- Using `eventlet` for:
  - Background task execution (`socketio.start_background_task()`)
  - Cooperative multitasking (`eventlet.sleep(0)`)
  - WebSocket async mode (`async_mode='eventlet'`)

### Migration Difficulty: ⭐⭐⭐ **MODERATE**

### Why Eventlet is Used
1. **WebSocket Support**: Real-time training progress updates
2. **Background Tasks**: Non-blocking neural network training
3. **Flask-SocketIO**: Requires an async backend

### Solution Options

#### Option A: Migrate to Gevent (Easiest)
**Difficulty**: Easy  
**Time**: 2-3 hours  
**Status**: ✅ Active maintenance, LTS

**Changes Required**:
```python
# In api_server.py
import gevent

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',  # Change from 'eventlet'
    logger=True,
    engineio_logger=True
)

# Replace eventlet.sleep(0) with:
gevent.sleep(0)
```

**requirements.txt**:
```
gevent>=23.9.1  # Replace eventlet
```

**Pros**:
- Drop-in replacement for eventlet
- Well-maintained (active development)
- Similar greenlet-based architecture
- Flask-SocketIO natively supports it

**Cons**:
- Still greenlet-based (not true async/await)
- Less popular than asyncio

#### Option B: Migrate to AsyncIO + Quart (Modern)
**Difficulty**: Moderate-Hard  
**Time**: 6-10 hours  
**Status**: ✅ Modern Python standard

**Changes Required**:
- Replace Flask with Quart (async Flask)
- Replace Flask-SocketIO with python-socketio with asyncio
- Convert all route handlers to async/await
- Use asyncio tasks instead of background threads

**Example**:
```python
from quart import Quart
import socketio

app = Quart(__name__)
sio = socketio.AsyncServer(async_mode='asgi')

@app.route('/api/networks/<network_id>/train', methods=['POST'])
async def train_network(network_id: str):
    # Start background task
    asyncio.create_task(
        train_network_task(network_id, job_id, ...)
    )
    return jsonify({...}), 202

async def train_network_task(...):
    # Async training loop
    await sio.emit('training_update', data)
```

**Pros**:
- Modern Python standard library
- Native async/await syntax
- Better performance for I/O-bound operations
- Future-proof

**Cons**:
- Requires rewriting all routes
- More complex concurrency model
- Learning curve if unfamiliar with async/await

#### Option C: Threading + Separate WebSocket Process
**Difficulty**: Moderate  
**Time**: 4-6 hours

**Architecture**:
- Flask API with standard threading
- Separate WebSocket server process
- Redis/message queue for communication

**Pros**:
- Simplest for the Flask API
- Clear separation of concerns
- Easy to scale horizontally

**Cons**:
- Requires Redis or similar
- More moving parts
- Added deployment complexity

---

## 3. Recommended Migration Path

### Phase 1: Quick Win (30 minutes)
✅ **Convert MNIST to NPZ format**
- Eliminates NumPy warning
- No API changes
- Minimal testing required

### Phase 2: Async Backend (2-3 hours)
✅ **Migrate Eventlet → Gevent**
- Minimal code changes
- Drop-in replacement
- Active LTS support
- Fully compatible with Flask-SocketIO

### Phase 3: Future (Optional, 6-10 hours)
⏳ **Consider AsyncIO Migration**
- When Python 3.12+ is baseline
- If performance becomes critical
- If expanding async operations

---

## 4. Implementation Plan

### Immediate Action Items

#### Step 1: Convert MNIST Data (30 min)
```python
# Create conversion script: scripts/convert_mnist_to_npz.py
import pickle
import gzip
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Load old format
with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train, val, test = u.load()

# Save as NPZ
np.savez_compressed(
    'data/mnist.npz',
    train_images=train[0],
    train_labels=train[1],
    val_images=val[0],
    val_labels=val[1],
    test_images=test[0],
    test_labels=test[1]
)
print("✅ Conversion complete!")
```

#### Step 2: Update Loader (15 min)
Update `mnist_loader.py` to use NPZ format

#### Step 3: Test (15 min)
Run full test suite to verify data loads correctly

#### Step 4: Migrate to Gevent (2 hours)
1. Update requirements.txt
2. Change imports in api_server.py
3. Update SocketIO initialization
4. Replace eventlet.sleep() calls
5. Test WebSocket functionality
6. Deploy

---

## 5. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Data conversion breaks training | Medium | Keep backup of original file, test thoroughly |
| Gevent incompatibility | Low | Flask-SocketIO officially supports gevent |
| WebSocket client issues | Low | No client changes needed |
| Performance regression | Low | Gevent has similar performance to eventlet |

---

## 6. Testing Checklist

- [ ] All unit tests pass
- [ ] MNIST data loads correctly
- [ ] Network training works
- [ ] WebSocket updates received
- [ ] Progress callbacks fire correctly
- [ ] Multiple concurrent training jobs work
- [ ] Network persistence works
- [ ] API endpoints respond correctly

---

## 7. Rollback Plan

1. Keep original `mnist.pkl.gz` file as backup
2. Git tag before migration: `git tag pre-gevent-migration`
3. If issues arise:
   ```bash
   git revert <commit-hash>
   pip install eventlet
   ```

---

## 8. Cost-Benefit Analysis

### Benefits
- ✅ No deprecation warnings
- ✅ Active LTS support (gevent)
- ✅ Better NumPy compatibility
- ✅ Future-proof codebase
- ✅ Cleaner, more maintainable

### Costs
- ⏱️ 6-10 hours development time
- 🧪 Testing and validation
- 📚 Minor documentation updates
- 🚀 Deployment coordination

### Verdict
**RECOMMENDED** - The benefits far outweigh the costs, especially considering:
- Eventlet is officially deprecated (maintenance only)
- NumPy warnings will get worse over time
- Migration difficulty is only moderate
- Gevent provides a clear upgrade path

---

## 9. Alternative: Do Nothing

### If you keep current setup:
- ⚠️ Warnings will persist (suppressed)
- ⚠️ Eventlet may break in future Python versions
- ⚠️ NumPy pickle format may become unsupported
- ⚠️ Technical debt accumulates

### When to act:
- ⏰ Before Python 3.15+ (eventlet may not work)
- ⏰ Before NumPy 3.0 (pickle format may break)
- ⏰ Before adding new async features

---

## 10. Conclusion

**Recommendation**: Migrate in two phases over 1-2 development sessions

1. **This week**: Convert MNIST to NPZ (30 min)
2. **Next sprint**: Migrate to Gevent (2-3 hours)

**Total effort**: ~3-4 hours for a clean, modern, maintainable codebase

The migration is **straightforward** and **low-risk** with clear benefits for long-term maintainability.
