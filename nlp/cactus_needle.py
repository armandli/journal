import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    import needle

    from typing import Literal
    from typing import Annotated
    from pydantic import BaseModel
    import json


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    ### simple usage: needle choose the right python function with the right argument to call, return result
    return


@app.function
@needle.tool
def get_weather(city: str):
    "Get the current weather for a city."
    return {"city": city, "temp_c": 27, "sky": "clear"}


@app.cell
def _():
    agent1 = needle.Needle(tools=[get_weather])
    print(agent1.run("what's it like in Lagos right now?")["results"])
    return


@app.cell
def _():
    ### medium usage: describe argument offering choices
    return


@app.function
@needle.tool
def set_thermostat(temperature: int, mode: Literal["heat", "cool", "auto"] = "auto"):
    """Set the thermostat.

    Args:
        temperature: target temperature in Celsius
        mode: heating strategy to use
    """
    return {"temperature": temperature, "mode": mode}


@app.cell
def _():
    agent2 = needle.Needle(tools=[set_thermostat])
    agent2.run("make it 21 and cool the room")
    return


@app.cell
def _():
    ### advanced usage: use needle.Field
    return


@app.function
@needle.tool
def send_money(
    amount: Annotated[float, needle.Field(gt=0, le=10000, description="USD, up to 10,000")],
    to:     Annotated[str,   needle.Field(pattern=r"^@[a-z0-9_]+$", description="recipient handle")],
    memo:   Annotated[str,   needle.Field(max_length=80)] = "",
):
    "Send money to a handle."
    return {"sent": amount, "to": to}


@app.cell
def _():
    return


@app.cell
def _():
    ### extraction
    return


@app.class_definition
class Invoice(BaseModel):
    vendor: str
    total: float
    due_date: str


@app.cell
def _():
    invoice = needle.extract("Invoice from Acme Corp, $1,200.00, due 2026-09-01", Invoice)
    print(invoice.vendor, invoice.total)
    return


@app.cell
def _():
    ### by hand
    return


@app.function
def set_lights(room, on, brightness):
    print(f"set lights on {room} brightness {brightness}")
    return {"room": room, "on": on, "brightness": brightness}


@app.cell
def _():
    tools = [{
        "name": "set_lights",
        "description": "Turn a room's lights on or off and set brightness",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {"type": "string", "description": "which room to control"},
                "on": {"type": "boolean"},
                "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["room", "on"],
        },
    }]
    return (tools,)


@app.cell
def _(tools):
    agent3 = needle.Needle(tools=tools)
    return (agent3,)


@app.cell
def _(agent3):
    response = agent3.complete("dim the living room to 30")
    print(response)
    if response["type"] == "call":
        result = set_lights(**response["function_calls"][0]["arguments"])
        response = agent3.complete(json.dumps(result))
        print(response)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
