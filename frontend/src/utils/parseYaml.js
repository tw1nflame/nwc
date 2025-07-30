import yaml from 'js-yaml';

export function parseYamlConfig(yamlText) {
  try {
    return yaml.load(yamlText);
  } catch (e) {
    console.error('YAML parse error:', e);
    return {};
  }
}
