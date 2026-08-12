import asyncio
from worldathletics import WorldAthletics

async def test():
    wa = WorldAthletics()
    
    # Test 1: fetch top 5 men's 100m for 2025
    print("=== Testing top list fetch ===")
    result = await wa.get_top_list(
        event_name="100 Metres",
        gender="men",
        environment="outdoor",
        year=2025,
        limit=5
    )
    print(type(result))
    print(result)

asyncio.run(test())