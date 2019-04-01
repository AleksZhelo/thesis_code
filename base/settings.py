import configparser
import ast
import os

from typing import Any


class Settings(object):

    def __init__(self, ini_file, sections=None):
        self.read_settings(ini_file, sections)

    def read_settings(self, ini_file, sections=None):
        """

        :param ini_file: the settings ini file
        :param sections: single or multiple section names, or None to read every section
        """

        config = configparser.ConfigParser()
        with open(ini_file) as f:
            try:
                config.read_file(f)
            except IOError as e:
                raise SettingsException(str(e))

        loaded_master = False

        if config.has_option('meta', 'master_settings_file'):
            master_path = Settings._parse_value(config.get('meta', 'master_settings_file'))
            master_path = master_path if os.path.isabs(master_path) else \
                os.path.join(os.path.dirname(str(ini_file)), master_path)
            self.read_settings(master_path, sections)
            loaded_master = True

        if sections is None:
            sections = config.sections()

        if isinstance(sections, str):
            setattr(self, sections, Settings._read_section(config, ini_file, sections, loaded_master))
        else:
            for section in sections:
                setattr(self, section, Settings._read_section(config, ini_file, section, loaded_master))

    @staticmethod
    def _read_section(config, ini_file, section, loaded_master):
        section_obj = Section()
        if not config.has_section(section):
            if not loaded_master:
                raise SettingsException('Section {0} not found in {1}'.format(section, ini_file))
        else:
            for opt in config.options(section):
                setattr(section_obj, opt, Settings._parse_value(config.get(section, opt)))
        return section_obj

    @staticmethod
    def _parse_value(value: str) -> Any:
        stripped = value.strip()
        if stripped == 'false':
            return False
        elif stripped == 'true':
            return True
        elif stripped.startswith("'") and stripped.endswith("'"):
            return stripped[1:-1]
        else:
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return stripped

    def __str__(self):
        lines = []
        for var in vars(self):
            value = getattr(self, var)
            if isinstance(value, Section):
                lines.append('[{0}]'.format(var))
                lines.append(str(value))
        return '\n'.join(lines)


class Section(object):
    def __str__(self):
        lines = []
        for var in vars(self):
            lines.append('{0}: {1}'.format(var, getattr(self, var)))
        return '\n'.join(lines)


class SettingsException(Exception):
    pass


if __name__ == '__main__':
    sett = Settings('../acoustic_word_embeddings/configs/conf.ini', None)
    print(sett)
