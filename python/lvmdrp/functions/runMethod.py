# TODO: define find_calib(frame):
#   * extract frame type, spec, channel and exptime
#   * determine calibration frames needed: bias, dark, flatfield, fiberflat, arc
#   * look for closest in time master calibration
# TODO: define prepQuick_drp(spec, channel, exposure, mjd):
#   * read a quick configuration template
#   * find target frame(s) in DB
#   * match with calibration frames from DB
#   * update quick configuration template(s)
#   * return quick configuration .YAML
# TODO: define prepFull_drp(spec, channel, exposure, mjd):
#   * read a full configuration template
#   * find target frame(s) in DB
#   * match with calibration frames
#   * update full configuration template(s)
#   * return full configuration .YAML
# TODO: define fromConfig_drp(config, **registered_modules):
#   * read config
#   * parse each DRP step in config (match config.steps to each module.step in registered_modules)
#   * run each DRP step