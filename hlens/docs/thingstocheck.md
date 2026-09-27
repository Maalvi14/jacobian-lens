Good catch to want the ratio check tightened up like that. Let me walk through the reasoning again more carefully first, since I want to be honest about what's actually guaranteed here versus what we're just observing empirically.

I traced through why the ratio comes out this consistent: v_t is built from u_t, and u_t is literally the gradient of unembed's t-th logit evaluated at the single fixed point h_ref. Because that gradient is taken at one shared point for every token, the norm's local linearization baked into u_t is the same linear map for every t — only the final lm_head weight row differs between tokens. So the whole "does the ratio match across tokens" question reduces to "does the norm's local-linear approximation at h_ref happen to behave proportionally to the real nonlinear norm, evaluated at this specific h" — which isn't something guaranteed to be exactly constant for an arbitrary h_ref and arbitrary h in general (it'd only be exactly constant if h pointed in the same direction as h_ref, which it doesn't here). So the near-378.2x-for-all-three consistency we saw is closer to "very good in this small toy case" than "mathematically guaranteed" — I don't want to write a test that quietly assumes it's exact when it isn't provably so.

Given that, the honest version of this test checks "ratios are consistent within some tolerance" (not bitwise-equal), and strengthens the check by trying a few different random h's — for each h, its own three token-ratios should cluster together, even though the ratio itself is expected to differ from one h to the next (it's h-dependent scale.)

```
(jlens) mv@MANUEL-VILLANUEVA2 jacobian-lens % pytest tests/test_jspace.py -s -v                      
========================================================================= test session starts ==========================================================================
platform darwin -- Python 3.12.11, pytest-9.0.3, pluggy-1.6.0 -- /Users/mv/repos/Anthropic/jacobian-lens/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /Users/mv/repos/Anthropic/jacobian-lens
configfile: pyproject.toml
plugins: anyio-4.13.0
collected 2 items                                                                                                                                                      

tests/test_jspace.py::test_lens_vectors_matches_lens_logit token 3: predicted=-53.5164  actual=-0.1415
token 7: predicted=-23.9759  actual=-0.0634
token 12: predicted=148.0138  actual=0.3913
PASSED
tests/test_jspace.py::test_lens_vectors_ratio_consistent_across_tokens PASSED
```
