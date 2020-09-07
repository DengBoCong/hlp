from configparser import ConfigParser
import json
import xml.dom.minidom as xl


def get_config_json(config_file='main.json'):
    with open(config_file) as file:
        return json.load(file)


def get_config_xml(config_file='main.xml'):
    dom = xl.parse(config_file)
    return dom.documentElement


def get_config_ini(config_file='main.ini'):
    parser = ConfigParser()
    parser.read(config_file)
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_floats + _conf_strings)
