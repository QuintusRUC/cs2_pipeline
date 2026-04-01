import steam_auth
import inspect

print("steam_auth module path:", steam_auth.__file__)
print("\n--- steam_auth.py contents (what Python sees) ---\n")
print(inspect.getsource(steam_auth))
print("\n--- attributes in steam_auth ---")
print([name for name in dir(steam_auth) if "steam" in name or "session" in name or "make" in name])
