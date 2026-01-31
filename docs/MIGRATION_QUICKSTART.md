# Quick Start: Migration to Modern Dependencies

This guide will help you migrate from deprecated dependencies to modern, LTS alternatives in just a few hours.

## TL;DR

**Difficulty**: ⭐⭐⭐ Moderate (3/5)  
**Time**: 3-4 hours total  
**Risk**: Low (easy rollback)  
**Benefit**: No warnings, future-proof, modern stack

---

## Step-by-Step Migration

### Phase 1: MNIST Data Format (30 minutes)

#### 1. Convert the data file
```bash
# Run the conversion script
python scripts/convert_mnist_to_npz.py
```

This will:
- ✅ Convert `mnist.pkl.gz` → `mnist.npz`
- ✅ Verify the conversion
- ✅ Create a backup
- ✅ Show next steps

#### 2. Update the loader

**Option A**: Replace the file (recommended)
```bash
# Backup the old loader
mv src/mnist_loader.py src/mnist_loader_legacy.py

# Use the new NPZ loader
mv src/mnist_loader_npz.py src/mnist_loader.py
```

**Option B**: Manual edit
Update `src/mnist_loader.py` to use NPZ format (see `mnist_loader_npz.py` for reference)

#### 3. Test
```bash
pytest
```

All 56 tests should pass! ✅

---

### Phase 2: Migrate Eventlet → Gevent (2-3 hours)

#### 1. Update dependencies
```bash
# Edit requirements.txt - replace:
eventlet

# With:
gevent>=23.9.1
```

#### 2. Install new dependency
```bash
pip install gevent>=23.9.1
pip uninstall eventlet
```

#### 3. Update api_server.py

**Replace these imports:**
```python
# OLD
import eventlet

# NEW
import gevent
```

**Update SocketIO initialization:**
```python
# OLD
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    ...
)

# NEW
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    ...
)
```

**Replace sleep calls:**
```python
# OLD
eventlet.sleep(0)

# NEW
gevent.sleep(0)
```

**Remove warning filters:**
```python
# DELETE these lines (no longer needed):
warnings.filterwarnings('ignore', category=DeprecationWarning)
import eventlet
warnings.filterwarnings('default', category=DeprecationWarning)
```

#### 4. Test everything
```bash
# Run all tests
pytest -v

# Test WebSocket manually
python src/api_server.py
# In another terminal:
curl http://localhost:8000/api/status
```

#### 5. Commit
```bash
git add .
git commit -m "Migrate from eventlet to gevent for LTS support"
```

---

## Verification Checklist

After migration, verify:

- [ ] All 56 tests pass
- [ ] No deprecation warnings in test output
- [ ] Server starts without errors
- [ ] Can create a network via API
- [ ] Can train a network
- [ ] WebSocket updates are received
- [ ] Can test network with examples
- [ ] Can save/load networks

---

## Rollback Instructions

If something goes wrong:

### Rollback Data Format
```bash
# Restore original pickle file
mv data/mnist.pkl.gz.backup data/mnist.pkl.gz

# Restore old loader
mv src/mnist_loader_legacy.py src/mnist_loader.py
```

### Rollback Gevent
```bash
# Revert the commit
git revert HEAD

# Reinstall eventlet
pip uninstall gevent
pip install eventlet
```

---

## Alternative: Automated Migration Script

Want to automate the whole process? Run:

```bash
# TODO: Create this script if you want full automation
python scripts/migrate_to_modern_stack.py
```

---

## What Changes for End Users?

**Nothing!** 🎉

The API remains exactly the same:
- Same endpoints
- Same request/response formats
- Same WebSocket protocol
- Same functionality

This is purely an internal upgrade.

---

## Performance Impact

**Expected changes:**
- ✅ Slightly faster MNIST loading (NPZ is more efficient)
- ≈ Same WebSocket performance (gevent ≈ eventlet)
- ✅ No deprecation warning overhead

---

## Support

If you run into issues:

1. Check `docs/MIGRATION_ASSESSMENT.md` for detailed information
2. Review the test output for specific errors
3. Use the rollback instructions above
4. File an issue with:
   - Python version
   - Error message
   - Steps to reproduce

---

## FAQ

**Q: Will this break my frontend?**  
A: No, the API is identical.

**Q: Do I need to retrain my networks?**  
A: No, saved networks are not affected.

**Q: Can I migrate in stages?**  
A: Yes! Do Phase 1 first, test, then do Phase 2.

**Q: What about Python 2?**  
A: You're already on Python 3.14+, so no concerns there.

**Q: Is gevent actively maintained?**  
A: Yes! Latest version is from 2023, with active development.

---

## After Migration

Once complete, you'll have:
- ✅ Zero deprecation warnings
- ✅ Modern, supported dependencies
- ✅ Future-proof codebase
- ✅ Better performance
- ✅ Cleaner code

Enjoy your modern stack! 🚀
