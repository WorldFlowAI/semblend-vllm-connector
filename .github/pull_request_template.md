## Summary

## Testing

- [ ] `make check`

## Connector Safety

- [ ] Exact prefix-cache semantics remain unchanged.
- [ ] Positive matched tokens are returned only when KV can be loaded.
- [ ] Non-identical semantic KV is not published into the exact prefix cache.
- [ ] User-visible behavior or config changes are documented.
