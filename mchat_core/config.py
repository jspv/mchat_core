from dynaconf import Dynaconf

default_files = ["settings.toml", ".secrets.toml"]


def get_settings(
    settings_files: list[str] = None,
) -> Dynaconf:
    if settings_files is None:
        settings_files = default_files
    return Dynaconf(
        envvar_prefix=False,
        ignore_unknown_envvars=True,
        settings_files=settings_files,
    )


# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
