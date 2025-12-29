// Chrono ground-truth CLI for Newton unit tests.
//
// This executable is intentionally small and only depends on Chrono. It reads a JSON request from stdin and prints a
// JSON response to stdout. It is used by Python unit tests to compare Newton implementations against Chrono outputs.

#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "chrono/functions/ChFunctionSineStep.h"
#include "chrono/utils/ChUtils.h"

#include "chrono_vehicle/ChTerrain.h"
#include "chrono_vehicle/wheeled_vehicle/ChTire.h"
#include "chrono_vehicle/wheeled_vehicle/tire/FialaTire.h"
#include "chrono_vehicle/utils/ChUtilsJSON.h"

#include "chrono_thirdparty/rapidjson/document.h"
#include "chrono_thirdparty/rapidjson/stringbuffer.h"
#include "chrono_thirdparty/rapidjson/writer.h"

namespace rj = rapidjson;

namespace {

std::string ReadAllStdin() {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    return ss.str();
}

[[noreturn]] void Throw(const std::string& msg) {
    throw std::runtime_error(msg);
}

double GetNumber(const rj::Value& v, const char* key) {
    if (!v.HasMember(key)) {
        Throw(std::string("missing key: ") + key);
    }
    const auto& x = v[key];
    if (!x.IsNumber()) {
        Throw(std::string("expected number for key: ") + key);
    }
    return x.GetDouble();
}

bool GetBool(const rj::Value& v, const char* key, bool default_val) {
    if (!v.HasMember(key)) {
        return default_val;
    }
    const auto& x = v[key];
    if (!x.IsBool()) {
        Throw(std::string("expected bool for key: ") + key);
    }
    return x.GetBool();
}

std::string GetString(const rj::Value& v, const char* key) {
    if (!v.HasMember(key)) {
        Throw(std::string("missing key: ") + key);
    }
    const auto& x = v[key];
    if (!x.IsString()) {
        Throw(std::string("expected string for key: ") + key);
    }
    return x.GetString();
}

chrono::ChVector3d GetVec3(const rj::Value& v, const char* key) {
    if (!v.HasMember(key)) {
        Throw(std::string("missing key: ") + key);
    }
    const auto& a = v[key];
    if (!a.IsArray() || a.Size() != 3) {
        Throw(std::string("expected vec3 array for key: ") + key);
    }
    return chrono::ChVector3d(a[0u].GetDouble(), a[1u].GetDouble(), a[2u].GetDouble());
}

void AddVec3(rj::Value& out_obj, rj::Document::AllocatorType& alloc, const char* key, const chrono::ChVector3d& v) {
    rj::Value arr(rj::kArrayType);
    arr.PushBack(v.x(), alloc);
    arr.PushBack(v.y(), alloc);
    arr.PushBack(v.z(), alloc);
    out_obj.AddMember(rj::Value().SetString(key, alloc), arr, alloc);
}

// Expose protected ChFialaTire functionality for testing.
class ExposedFialaTire final : public chrono::vehicle::FialaTire {
  public:
    static ExposedFialaTire FromFile(const std::string& filename) {
        rapidjson::Document d;
        chrono::vehicle::ReadFileJSON(filename, d);
        if (d.IsNull()) {
            Throw(std::string("failed to read tire JSON: ") + filename);
        }
        return ExposedFialaTire(d);
    }

    explicit ExposedFialaTire(const rapidjson::Document& d) : chrono::vehicle::FialaTire(d) {}

    void SetMu(double mu) { m_mu = mu; }

    double GetMu0() const { return m_mu_0; }

    double GetRollingResistance() const { return m_rolling_resistance; }

    double GetUnloadedRadius() const { return m_unloaded_radius; }

    double GetWidth() const { return m_width; }

    void PatchForces(double& fx, double& fy, double& mz, double kappa, double alpha, double fz) {
        FialaPatchForces(fx, fy, mz, kappa, alpha, fz);
    }
};

// Access helper for protected static utilities on ChTire.
class TireAccess : public chrono::vehicle::ChTire {
  public:
    using chrono::vehicle::ChTire::ChTire;
    static void ConstructAreaDepthTablePublic(double disc_radius, chrono::ChFunctionInterp& areaDep) {
        chrono::vehicle::ChTire::ConstructAreaDepthTable(disc_radius, areaDep);
    }

    static bool DiscTerrainCollisionPublic(chrono::vehicle::ChTire::CollisionType method,
                                          const chrono::vehicle::ChTerrain& terrain,
                                          const chrono::ChVector3d& disc_center,
                                          const chrono::ChVector3d& disc_normal,
                                          double disc_radius,
                                          double width,
                                          const chrono::ChFunctionInterp& areaDep,
                                          chrono::ChCoordsys<>& contact,
                                          double& depth,
                                          float& mu) {
        return chrono::vehicle::ChTire::DiscTerrainCollision(method, terrain, disc_center, disc_normal, disc_radius, width,
                                                            areaDep, contact, depth, mu);
    }
};

class AnalyticTerrain final : public chrono::vehicle::ChTerrain {
  public:
    enum class Type { PLANE, SINUSOID };

    static AnalyticTerrain MakePlane(const chrono::ChVector3d& point,
                                     const chrono::ChVector3d& normal,
                                     float mu) {
        AnalyticTerrain t;
        t.m_type = Type::PLANE;
        t.m_point = point;
        t.m_normal = normal;
        t.m_normal.Normalize();
        t.m_mu = mu;
        return t;
    }

    static AnalyticTerrain MakeSinusoid(double base,
                                        double amp,
                                        double freq,
                                        float mu) {
        AnalyticTerrain t;
        t.m_type = Type::SINUSOID;
        t.m_base = base;
        t.m_amp = amp;
        t.m_freq = freq;
        t.m_mu = mu;
        return t;
    }

    double GetHeight(const chrono::ChVector3d& loc) const override {
        if (m_type == Type::PLANE) {
            const double nz = m_normal.z();
            if (std::abs(nz) < 1e-12) {
                // Not a height field (vertical plane). Undefined for our purposes.
                return m_point.z();
            }
            const double dx = loc.x() - m_point.x();
            const double dy = loc.y() - m_point.y();
            const double dz = -(m_normal.x() * dx + m_normal.y() * dy) / nz;
            return m_point.z() + dz;
        }

        // z = base + amp*sin(freq*x)*sin(freq*y)
        return m_base + m_amp * std::sin(m_freq * loc.x()) * std::sin(m_freq * loc.y());
    }

    chrono::ChVector3d GetNormal(const chrono::ChVector3d& loc) const override {
        if (m_type == Type::PLANE) {
            return m_normal;
        }

        // z = f(x,y); normal ~ (-df/dx, -df/dy, 1)
        // NOTE (Lukas): F(x,y,z) = f(x,y) - z = 0 defines the surface. ∇F is a normal to the surface.
        // consider any curve r  on surface F(r(t), then clearly d​F/dt(r(t))=∇F(r(t))⋅r′(t) = 0, r′(t) is any tangent vector.
        // In simple terms: Surface is defined as a level curve. Gradient is always orthogonal to level curves. 
        const double sx = std::sin(m_freq * loc.x());
        const double sy = std::sin(m_freq * loc.y());
        const double cx = std::cos(m_freq * loc.x());
        const double cy = std::cos(m_freq * loc.y());
        const double dfdx = m_amp * m_freq * cx * sy;
        const double dfdy = m_amp * m_freq * sx * cy;
        chrono::ChVector3d n(-dfdx, -dfdy, 1.0);
        n.Normalize();
        return n;
    }

    float GetCoefficientFriction(const chrono::ChVector3d& loc) const override {
        (void)loc;
        return m_mu;
    }

    void GetProperties(const chrono::ChVector3d& loc,
                       double& height,
                       chrono::ChVector3d& normal,
                       float& friction) const override {
        height = GetHeight(loc);
        normal = GetNormal(loc);
        friction = GetCoefficientFriction(loc);
    }

  private:
    Type m_type = Type::PLANE;
    chrono::ChVector3d m_point = chrono::ChVector3d(0, 0, 0);
    chrono::ChVector3d m_normal = chrono::ChVector3d(0, 0, 1);
    float m_mu = 0.8f;

    double m_base = 0.0;
    double m_amp = 0.0;
    double m_freq = 1.0;
};

AnalyticTerrain ParseTerrain(const rj::Value& req) {
    if (!req.HasMember("terrain")) {
        Throw("missing key: terrain");
    }
    const auto& t = req["terrain"];
    if (!t.IsObject()) {
        Throw("expected object: terrain");
    }
    const std::string type = GetString(t, "type");
    const float mu = static_cast<float>(t.HasMember("mu") ? GetNumber(t, "mu") : 0.8);

    if (type == "plane") {
        // Supported forms:
        // - {"type":"plane","height":0.0,"mu":0.8}
        // - {"type":"plane","point":[...],"normal":[...],"mu":0.8}
        if (t.HasMember("height")) {
            const double h = GetNumber(t, "height");
            return AnalyticTerrain::MakePlane(chrono::ChVector3d(0, 0, h), chrono::ChVector3d(0, 0, 1), mu);
        }
        const auto p = GetVec3(t, "point");
        const auto n = GetVec3(t, "normal");
        return AnalyticTerrain::MakePlane(p, n, mu);
    }

    if (type == "sinusoid") {
        const double base = GetNumber(t, "base");
        const double amp = GetNumber(t, "amp");
        const double freq = GetNumber(t, "freq");
        return AnalyticTerrain::MakeSinusoid(base, amp, freq, mu);
    }

    Throw(std::string("unsupported terrain.type: ") + type);
}

chrono::vehicle::ChTire::CollisionType ParseCollisionType(const std::string& s) {
    if (s == "single_point") {
        return chrono::vehicle::ChTire::CollisionType::SINGLE_POINT;
    }
    if (s == "four_points") {
        return chrono::vehicle::ChTire::CollisionType::FOUR_POINTS;
    }
    if (s == "envelope") {
        return chrono::vehicle::ChTire::CollisionType::ENVELOPE;
    }
    Throw(std::string("unsupported collision_type: ") + s);
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    rj::Document resp;
    resp.SetObject();
    auto& alloc = resp.GetAllocator();

    int exit_code = 0;
    try {
        const std::string req_str = ReadAllStdin();
        if (req_str.empty()) {
            Throw("empty request on stdin");
        }

        rj::Document req;
        req.Parse(req_str.c_str());
        if (req.HasParseError() || !req.IsObject()) {
            Throw("failed to parse JSON request (expected an object)");
        }

        const std::string cmd = GetString(req, "cmd");
        resp.AddMember("cmd", rj::Value().SetString(cmd.c_str(), alloc), alloc);

        if (cmd == "get_params") {
            const std::string tire_json = GetString(req, "tire_json");
            auto tire = ExposedFialaTire::FromFile(tire_json);

            resp.AddMember("unloaded_radius", tire.GetUnloadedRadius(), alloc);
            resp.AddMember("width", tire.GetWidth(), alloc);
            resp.AddMember("rolling_resistance", tire.GetRollingResistance(), alloc);
            resp.AddMember("mu0", tire.GetMu0(), alloc);
            resp.AddMember("ok", true, alloc);
        } else if (cmd == "normal_stiffness_force") {
            const std::string tire_json = GetString(req, "tire_json");
            const double depth = GetNumber(req, "depth");
            auto tire = ExposedFialaTire::FromFile(tire_json);

            resp.AddMember("fz_stiff", tire.GetNormalStiffnessForce(depth), alloc);
            resp.AddMember("ok", true, alloc);
        } else if (cmd == "normal_damping_force") {
            const std::string tire_json = GetString(req, "tire_json");
            const double depth = GetNumber(req, "depth");
            const double velocity = GetNumber(req, "velocity");
            auto tire = ExposedFialaTire::FromFile(tire_json);

            resp.AddMember("fz_damp", tire.GetNormalDampingForce(depth, velocity), alloc);
            resp.AddMember("ok", true, alloc);
        } else if (cmd == "normal_load") {
            // Match Chrono's ChFialaTire::Synchronize:
            // Fn = fstiff(depth) + fdamp(depth, -vel_z), clamped to Fn>=0.
            const std::string tire_json = GetString(req, "tire_json");
            const double depth = GetNumber(req, "depth");
            const double vel_z = GetNumber(req, "vel_z");
            auto tire = ExposedFialaTire::FromFile(tire_json);

            const double fz_stiff = tire.GetNormalStiffnessForce(depth);
            const double fz_damp = tire.GetNormalDampingForce(depth, -vel_z);
            double fz = fz_stiff + fz_damp;
            if (fz < 0.0) {
                fz = 0.0;
            }
            resp.AddMember("fz_stiff", fz_stiff, alloc);
            resp.AddMember("fz_damp", fz_damp, alloc);
            resp.AddMember("fz", fz, alloc);
            resp.AddMember("ok", true, alloc);
        } else if (cmd == "fiala_patch_forces") {
            const std::string tire_json = GetString(req, "tire_json");
            const double kappa = GetNumber(req, "kappa");
            const double alpha = GetNumber(req, "alpha");
            const double fz = GetNumber(req, "fz");
            const double mu_in = GetNumber(req, "mu");
            const bool clamp_mu = GetBool(req, "clamp_mu", true);

            auto tire = ExposedFialaTire::FromFile(tire_json);
            double mu = mu_in;
            if (clamp_mu) {
                chrono::ChClampValue(mu, 0.1, 1.0);
            }
            tire.SetMu(mu);

            double fx = 0.0;
            double fy = 0.0;
            double mz = 0.0;
            tire.PatchForces(fx, fy, mz, kappa, alpha, fz);

            resp.AddMember("mu", mu, alloc);
            resp.AddMember("fx", fx, alloc);
            resp.AddMember("fy", fy, alloc);
            resp.AddMember("mz", mz, alloc);
            resp.AddMember("ok", true, alloc);
        } else if (cmd == "rolling_resistance_moment") {
            // Match Chrono's ChFialaTire::Advance rolling resistance:
            // myStartUp = SineStep(|Vx|, vx_min=0.125 -> 0, vx_max=0.5 -> 1)
            // My = -myStartUp * rr * fz * sign(omega)
            const std::string tire_json = GetString(req, "tire_json");
            const double abs_vx = GetNumber(req, "abs_vx");
            const double fz = GetNumber(req, "fz");
            const double omega = GetNumber(req, "omega");
            auto tire = ExposedFialaTire::FromFile(tire_json);

            const double vx_min = 0.125;
            const double vx_max = 0.5;
            const double my_start = chrono::ChFunctionSineStep::Eval(abs_vx, vx_min, 0.0, vx_max, 1.0);
            const double My = -my_start * tire.GetRollingResistance() * fz * chrono::ChSignum(omega);

            resp.AddMember("my_startup", my_start, alloc);
            resp.AddMember("My", My, alloc);
            resp.AddMember("ok", true, alloc);
        } else if (cmd == "fiala_slip") {
            // Match Chrono's ChFialaTire slip definitions (Advance + Synchronize bookkeeping):
            //   vsx   = v_x - omega * r_eff
            //   vsy   = v_y
            //   kappa = -vsx / abs(v_x)      (if abs(v_x) != 0 else 0)
            //   alpha = atan2(v_y, abs(v_x)) (if abs(v_x) != 0 else 0)
            const double v_x = GetNumber(req, "v_x");
            const double v_y = GetNumber(req, "v_y");
            const double omega = GetNumber(req, "omega");
            const double r_eff = GetNumber(req, "r_eff");

            const double abs_vx = std::abs(v_x);
            const double abs_vt = std::abs(omega * r_eff);
            const double vsx = v_x - omega * r_eff;
            const double vsy = v_y;

            double kappa = 0.0;
            double alpha = 0.0;
            if (abs_vx != 0.0) {
                kappa = -vsx / abs_vx;
                alpha = std::atan2(vsy, abs_vx);
            }

            resp.AddMember("abs_vx", abs_vx, alloc);
            resp.AddMember("abs_vt", abs_vt, alloc);
            resp.AddMember("vsx", vsx, alloc);
            resp.AddMember("vsy", vsy, alloc);
            resp.AddMember("kappa", kappa, alloc);
            resp.AddMember("alpha", alpha, alloc);
            resp.AddMember("ok", true, alloc);
        } else if (cmd == "disc_terrain_collision") {
            const std::string method_str = GetString(req, "collision_type");
            const auto method = ParseCollisionType(method_str);

            const chrono::ChVector3d disc_center = GetVec3(req, "disc_center");
            chrono::ChVector3d disc_normal = GetVec3(req, "disc_normal");
            if (disc_normal.Length2() < 1e-20) {
                Throw("disc_normal is near zero");
            }
            disc_normal.Normalize();

            const double disc_radius = GetNumber(req, "disc_radius");
            const double width = GetNumber(req, "width");

            const auto terrain = ParseTerrain(req);

            chrono::ChCoordsys<> contact;
            double depth = 0.0;
            float mu = 0.0f;
            chrono::ChFunctionInterp area_dep;
            TireAccess::ConstructAreaDepthTablePublic(disc_radius, area_dep);

            const bool in_contact = TireAccess::DiscTerrainCollisionPublic(
                method, terrain, disc_center, disc_normal, disc_radius, width, area_dep, contact, depth, mu);

            resp.AddMember("in_contact", in_contact, alloc);
            resp.AddMember("depth", depth, alloc);
            resp.AddMember("mu", static_cast<double>(mu), alloc);

            rj::Value contact_obj(rj::kObjectType);
            AddVec3(contact_obj, alloc, "pos", contact.pos);
            chrono::ChMatrix33<> A(contact.rot);
            AddVec3(contact_obj, alloc, "x_axis", A.GetAxisX());
            AddVec3(contact_obj, alloc, "y_axis", A.GetAxisY());
            AddVec3(contact_obj, alloc, "z_axis", A.GetAxisZ());
            resp.AddMember("contact", contact_obj, alloc);
            resp.AddMember("ok", true, alloc);
        } else {
            Throw(std::string("unsupported cmd: ") + cmd);
        }
    } catch (const std::exception& e) {
        resp.AddMember("ok", false, alloc);
        resp.AddMember("error", rj::Value().SetString(e.what(), alloc), alloc);
        exit_code = 1;
    }

    rj::StringBuffer buffer;
    rj::Writer<rj::StringBuffer> writer(buffer);
    resp.Accept(writer);
    std::cout << buffer.GetString() << std::endl;

    return exit_code;
}
