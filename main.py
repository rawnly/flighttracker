import asyncio
import math
import numpy
from airportsdata import Airport, load as get_airports
from fast_flights import (
    FlightData,
    Passengers,
    Result,
    create_filter,
    get_flights_from_filter,
    search_airport,
)
from typing import List
from fastapi import FastAPI

# from fastapi_mcp import FastApiMCP
from pydantic import BaseModel
from fastmcp import FastMCP, Client

app = FastAPI()
# mcp = FastApiMCP(
#     app
#     # name="FlightTracker MCP",
#     # description="MCP server for the flight tracker api",
#     # describe_full_response_schema=True,  # Describe the full response JSON-schema instead of just a response example
#     # describe_all_responses=True,  # Describe all the possible responses instead of just the success (2XX) response
# )

airports: dict[str, Airport] | None = None

# mcp.mount()


@app.post("/airports")
def list_airports(query: str | None = None):
    global airports
    if query:
        data = search_airport(query)
        return data

    if airports is None or not airports:
        airports = get_airports("IATA")

    return airports


class SearchPayload(BaseModel):
    short_response: bool = False
    departure_airport: str
    arrival_airports: List[str]
    departure_date: str
    return_date: str | None = None
    stops: int | None = None
    currency: str = ""
    only_best: bool = False

    def to_flight_data(self):
        for destination in self.arrival_airports:
            yield {
                "flight_data": [
                    FlightData(
                        date=self.departure_date,
                        to_airport=destination,
                        from_airport=self.departure_airport,
                        max_stops=self.stops,
                    ),
                    FlightData(
                        date=self.return_date or self.departure_date,
                        to_airport=self.departure_airport,
                        from_airport=destination,
                        max_stops=self.stops,
                    ),
                ],
                "departure": self.departure_airport,
                "arrival": destination,
                "dates": {
                    "departure": self.departure_date,
                    "return": self.return_date or self.departure_date,
                },
            }

    def validate_airports(self):
        global airports

        if airports is None or not airports:
            airports = get_airports("IATA")

        if self.departure_airport not in airports:
            raise ValueError(f"Invalid departure airport: {self.departure_airport}")

        for arrival in self.arrival_airports:
            if arrival not in airports:
                raise ValueError(f"Invalid arrival airport: {arrival}")


@app.post("/flights")
async def search_flights(payload: SearchPayload):
    """
    Search for flights from one location to another.

    Args:
        from (str): Departure location.
        to (List[str]): List of destination locations.
        date (str): Departure date in YYYY-MM-DD format.
        return_date (str, optional): Return date in YYYY-MM-DD format. Defaults to None.

    Returns:
        dict: A dictionary containing flight search results.
    """

    try:
        payload.validate_airports()
    except ValueError as e:
        return {"error": str(e)}

    data: dict[str, any] = {}

    for element in payload.to_flight_data():
        try:
            filter = create_filter(
                flight_data=element["flight_data"],
                trip="round-trip",
                passengers=Passengers(adults=1),
                seat="economy",
                max_stops=payload.stops,
            )

            result = get_flights_from_filter(
                filter=filter,
                currency=payload.currency,
            )

            flights = result.flights

            if payload.only_best:
                flights = []

                for flight in result.flights:
                    if flight.is_best:
                        flights.append(flight)

            price_stats = get_stats(result)

            data[element["arrival"]] = {
                "dates": element["dates"],
                "price": price_stats,
                "current_price": result.current_price,
                "departure": element["departure"],
                "arrival": element["arrival"],
                "flights": flights,
            }

            if payload.short_response:
                data[element["arrival"]].pop("flights")
        except Exception as e:
            print(
                f"Error fetching flights for {element['departure']} to {element['arrival']}: {e}"
            )
            data[element["arrival"]] = None

    return data


def get_stats(result: Result):
    """
    Calculate the average price of flights in a Result object.
    """

    # get price and remove currency (first character)
    prices = [
        int(f.price[1:].replace(",", "").replace(".", "")) for f in result.flights
    ]

    if not prices:
        return {
            "avg": 0,
            "low": 0,
            "high": 0,
        }

    avg = math.floor(numpy.median(prices))
    low = math.floor(min(prices))
    high = math.floor(max(prices))

    return {
        "avg": avg,
        "low": low,
        "high": high,
    }


@app.get("/")
def root():
    return {"message": "Welcome to the Flight Tracker API!"}


# Test your MCP server with a client
async def check_mcp(mcp: FastMCP):
    # List the components that were created
    tools = await mcp.get_tools()
    resources = await mcp.get_resources()
    templates = await mcp.get_resource_templates()

    print(f"{len(tools)} Tool(s): {', '.join([t.name for t in tools.values()])}")
    print(
        f"{len(resources)} Resource(s): {', '.join([r.name for r in resources.values()])}"  # type: ignore
    )
    print(
        f"{len(templates)} Resource Template(s): {', '.join([t.name for t in templates.values()])}"
    )

    return mcp


if __name__ == "__main__":
    mcp = FastMCP.from_fastapi(app=app)
    asyncio.run(check_mcp(mcp))
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp")
