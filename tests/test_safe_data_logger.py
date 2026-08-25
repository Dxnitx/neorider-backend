from safe_data_logger import parse_sensor_line


def test_parse_sensor_line():
    assert parse_sensor_line("1.25,Helmet,0.1,0.2,1.0,2.0,3.0,4.0") == (
        "helmet", [1.25, 0.1, 0.2, 1.0, 2.0, 3.0, 4.0]
    )
    assert parse_sensor_line("time,device,ax,ay,az,rx,ry,rz") is None
    assert parse_sensor_line("ESP32 connected") is None
    assert parse_sensor_line("1,chest,invalid,0,1,0,0,0") is None
