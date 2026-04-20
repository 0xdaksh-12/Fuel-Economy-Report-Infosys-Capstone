"""
SmartWay Vehicle Predictor
--------------------------
Loads the saved Decision Tree model and predicts whether a vehicle
qualifies as a SmartWay vehicle.

Usage:
    uv run predict                # uses built-in default example
"""

import argparse
import pathlib

import joblib
import pandas as pd

# Path to model relative to project root
_MODEL_PATH = pathlib.Path(__file__).parent.parent / "model" / "smartway_decision_tree_model.pkl"

LABEL_MAP = {0: "No", 1: "Yes", 2: "Elite"}


def predict(
    disp: float,
    cyl: int,
    ap: int,
    city_mpg: float,
    hwy_mpg: float,
    comb_mpg: float,
    ghg: int,
    co2: float,
    trans: str,
    drive: str,
    fuel: str,
    cert: str,
    veh_class: str
) -> str:
    model = joblib.load(_MODEL_PATH)
    
    # Categoricals need to be passed as mapped integers since the model used LabelEncoder
    vehicle = pd.DataFrame([{
        "Engine_Displacement_L": float(disp),
        "Engine_Cylinders": int(cyl),
        "Air_Pollution_Score": int(ap),
        "City_Mpg": float(city_mpg),
        "Hwy_Mpg": float(hwy_mpg),
        "Combined_Mpg": float(comb_mpg),
        "Greenhouse_Gas_Score": int(ghg),
        "Combined_Co2": float(co2),
        "Transmission_Type": int(trans),
        "Drive": int(drive),
        "Fuel": int(fuel),
        "Cert_Region": int(cert),
        "Veh_Class": int(veh_class),
    }])
    result = model.predict(vehicle)
    return LABEL_MAP.get(int(result[0]), str(result[0]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict SmartWay status for a vehicle."
    )
    parser.add_argument("--disp",  type=float, default=2.0,  help="Engine displacement (litres)")
    parser.add_argument("--cyl",   type=int,   default=4,    help="Number of cylinders")
    parser.add_argument("--ap",    type=int,   default=5,    help="Air Pollution Score (1-10)")
    parser.add_argument("--city_mpg", type=float, default=25.0, help="City MPG")
    parser.add_argument("--hwy_mpg", type=float, default=33.0, help="Highway MPG")
    parser.add_argument("--comb_mpg", type=float, default=28.0, help="Combined MPG")
    parser.add_argument("--ghg",   type=int,   default=5,    help="Greenhouse Gas Score (1-10)")
    parser.add_argument("--co2",   type=float, default=300.0, help="Combined CO2")
    parser.add_argument("--trans", type=int,   default=1, help="Transmission Type (int mapped)")
    parser.add_argument("--drive", type=int,   default=1, help="Drive wheels (int mapped)")
    parser.add_argument("--fuel",  type=int,   default=1, help="Fuel type (int mapped)")
    parser.add_argument("--cert",  type=int,   default=1, help="Cert Region (int mapped)")
    parser.add_argument("--veh_class", type=int, default=1, help="Vehicle Class (int mapped)")
    args = parser.parse_args()

    label = predict(
        disp=args.disp,
        cyl=args.cyl,
        ap=args.ap,
        city_mpg=args.city_mpg,
        hwy_mpg=args.hwy_mpg,
        comb_mpg=args.comb_mpg,
        ghg=args.ghg,
        co2=args.co2,
        trans=args.trans,
        drive=args.drive,
        fuel=args.fuel,
        cert=args.cert,
        veh_class=args.veh_class,
    )

    print(f"\n  SmartWay Prediction: {label}\n")


if __name__ == "__main__":
    main()
