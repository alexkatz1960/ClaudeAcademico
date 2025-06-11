# test.py
try:
    import integrations
    health = integrations.health_check()
    print(f"Status: {health['status']}")
    for component, status in health['components'].items():
        print(f"  {component}: {'✅' if status else '❌'}")
except Exception as e:
    print(f"Error: {e}")