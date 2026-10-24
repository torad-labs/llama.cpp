#pragma once
#include <map>
#include <string>
struct common_params;

// Pure resolution is also used by experiment tooling; explicit overrides are
// validated together before any process environment or allocation is changed.
std::map<std::string, std::string> common_glm53_profile_resolve(
    const std::string & name, const std::map<std::string, std::string> & overrides = {});
void common_glm53_profile_apply(common_params & params);
