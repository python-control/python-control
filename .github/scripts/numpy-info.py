from numpy.core._multiarray_umath import (
        __cpu_features__, __cpu_baseline__, __cpu_dispatch__)

features_found, features_not_found = [], []
for feature in __cpu_dispatch__:
    if __cpu_features__[feature]:
        features_found.append(feature)
    else:
        features_not_found.append(feature)

print("Supported SIMD extensions in this NumPy install:")
print("    baseline = %s" % (','.join(__cpu_baseline__)))
print("    found = %s" % (','.join(features_found)))
print("    not found = %s" % (','.join(features_not_found)))
