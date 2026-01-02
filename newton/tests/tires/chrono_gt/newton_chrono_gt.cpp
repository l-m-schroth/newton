// Chrono ground-truth CLI for Newton unit tests.
//
// This executable is intentionally small and only depends on Chrono. It reads a JSON request from stdin and prints a
// JSON response to stdout. It is used by Python unit tests to compare Newton implementations against Chrono outputs.

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>

#include "chrono/core/ChTypes.h"
#include "chrono/physics/ChSystemNSC.h"
#include "chrono/solver/ChIterativeSolverVI.h"

#include "chrono/functions/ChFunctionConst.h"
#include "chrono/functions/ChFunctionSine.h"
#include "chrono/functions/ChFunctionSineStep.h"
#include "chrono/utils/ChUtils.h"

#include "chrono_vehicle/ChTerrain.h"
#include "chrono_vehicle/wheeled_vehicle/ChTire.h"
#include "chrono_vehicle/wheeled_vehicle/test_rig/ChTireTestRig.h"
#include "chrono_vehicle/wheeled_vehicle/tire/FialaTire.h"
#include "chrono_vehicle/utils/ChUtilsJSON.h"

#include "chrono_thirdparty/rapidjson/document.h"
#include "chrono_thirdparty/rapidjson/stringbuffer.h"
#include "chrono_thirdparty/rapidjson/writer.h"

namespace rj = rapidjson;

namespace {

void ConfigureChronoTireTestRigSystem(chrono::ChSystemNSC& sys) {
    // Chrono reference:
    // - chrono/src/demos/vehicle/test_rigs/demo_VEH_TireTestRig.cpp (NSC default solver/integrator configuration)
    // - chrono/src/demos/SetChronoSolver.h::SetChronoSolver (BB solver parameters)
    //
    // Rationale: the tire test rig is sensitive to constraint drift during the settling phase. The Chrono demo sets an
    // iterative VI solver (BARZILAIBORWEIN) and EULER_IMPLICIT_LINEARIZED integrator. We mirror those settings here so
    // the ground-truth behavior matches the Chrono reference demo more closely.

    sys.SetSolverType(chrono::ChSolver::Type::BARZILAIBORWEIN);
    if (auto solver = std::dynamic_pointer_cast<chrono::ChIterativeSolverVI>(sys.GetSolver())) {
        solver->SetMaxIterations(100);
        solver->SetOmega(0.8);
        solver->SetSharpnessLambda(1.0);
    }

    sys.SetTimestepperType(chrono::ChTimestepper::Type::EULER_IMPLICIT_LINEARIZED);
}

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

std::array<double, 4> GetVec4(const rj::Value& v, const char* key) {
    if (!v.HasMember(key)) {
        Throw(std::string("missing key: ") + key);
    }
    const auto& a = v[key];
    if (!a.IsArray() || a.Size() != 4) {
        Throw(std::string("expected vec4 array for key: ") + key);
    }
    return {
        a[0u].GetDouble(),
        a[1u].GetDouble(),
        a[2u].GetDouble(),
        a[3u].GetDouble(),
    };
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

class HFieldTerrain final : public chrono::vehicle::ChTerrain {
  public:
    struct Params {
        std::array<double, 4> size = {1.0, 1.0, 1.0, 0.1};  // (x, y, z_top, z_bottom)
        int nrow = 0;
        int ncol = 0;
        std::vector<float> data;
        chrono::ChVector3d pos = chrono::ChVector3d(0, 0, 0);  // base plane origin
        float mu = 0.8f;
    };

    static HFieldTerrain Make(const Params& p) {
        HFieldTerrain t;
        t.m_size = p.size;
        t.m_nrow = p.nrow;
        t.m_ncol = p.ncol;
        t.m_data = p.data;
        t.m_pos = p.pos;
        t.m_mu = p.mu;
        return t;
    }

    double GetHeight(const chrono::ChVector3d& loc) const override {
        // MuJoCo reference: mujoco/src/engine/engine_collision_convex.c::mjc_ConvexHField (grid layout)
        // MuJoCo reference: mujoco/src/engine/engine_ray.c::mj_rayHfieldNormal (triangulation)
        if (m_nrow < 2 || m_ncol < 2 || m_data.empty()) {
            return m_pos.z();
        }

        const double size_x = m_size[0];
        const double size_y = m_size[1];
        const double size_z = m_size[2];  // z_top

        const double dx = (2.0 * size_x) / double(m_ncol - 1);
        const double dy = (2.0 * size_y) / double(m_nrow - 1);
        if (dx == 0.0 || dy == 0.0) {
            return m_pos.z();
        }

        const double x = loc.x() - m_pos.x();
        const double y = loc.y() - m_pos.y();

        double u = (x + size_x) / dx;
        double v = (y + size_y) / dy;
        u = chrono::ChClamp(u, 0.0, double(m_ncol - 1));
        v = chrono::ChClamp(v, 0.0, double(m_nrow - 1));

        int c = int(std::floor(u));
        int r = int(std::floor(v));
        if (c > m_ncol - 2) {
            c = m_ncol - 2;
        }
        if (r > m_nrow - 2) {
            r = m_nrow - 2;
        }

        const double tx = u - double(c);
        const double ty = v - double(r);

        auto h = [&](int rr, int cc) -> double { return double(m_data[rr * m_ncol + cc]) * size_z; };
        const double h00 = h(r, c);
        const double h10 = h(r, c + 1);
        const double h01 = h(r + 1, c);
        const double h11 = h(r + 1, c + 1);

        double z = 0.0;
        if (tx >= ty) {
            // tri1 (v00, v11, v10): weights w00=1-tx, w10=tx-ty, w11=ty
            z = (1.0 - tx) * h00 + (tx - ty) * h10 + ty * h11;
        } else {
            // tri2 (v00, v01, v11): weights w00=1-ty, w01=ty-tx, w11=tx
            z = (1.0 - ty) * h00 + (ty - tx) * h01 + tx * h11;
        }

        return m_pos.z() + z;
    }

    chrono::ChVector3d GetNormal(const chrono::ChVector3d& loc) const override {
        if (m_nrow < 2 || m_ncol < 2 || m_data.empty()) {
            return chrono::ChVector3d(0, 0, 1);
        }

        const double size_x = m_size[0];
        const double size_y = m_size[1];
        const double size_z = m_size[2];  // z_top

        const double dx = (2.0 * size_x) / double(m_ncol - 1);
        const double dy = (2.0 * size_y) / double(m_nrow - 1);
        if (dx == 0.0 || dy == 0.0) {
            return chrono::ChVector3d(0, 0, 1);
        }

        const double x = loc.x() - m_pos.x();
        const double y = loc.y() - m_pos.y();

        double u = (x + size_x) / dx;
        double v = (y + size_y) / dy;
        u = chrono::ChClamp(u, 0.0, double(m_ncol - 1));
        v = chrono::ChClamp(v, 0.0, double(m_nrow - 1));

        int c = int(std::floor(u));
        int r = int(std::floor(v));
        if (c > m_ncol - 2) {
            c = m_ncol - 2;
        }
        if (r > m_nrow - 2) {
            r = m_nrow - 2;
        }

        const double tx = u - double(c);
        const double ty = v - double(r);

        auto h = [&](int rr, int cc) -> double { return double(m_data[rr * m_ncol + cc]) * size_z; };
        const double h00 = h(r, c);
        const double h10 = h(r, c + 1);
        const double h01 = h(r + 1, c);
        const double h11 = h(r + 1, c + 1);

        chrono::ChVector3d n(0, 0, 1);
        if (tx >= ty) {
            // normal ~ cross(v10-v00, v11-v00)
            const double dz10 = h10 - h00;
            const double dz11 = h11 - h00;
            n = chrono::ChVector3d(-dz10 * dy, dx * (dz10 - dz11), dx * dy);
        } else {
            // normal ~ cross(v11-v00, v01-v00)
            const double dz01 = h01 - h00;
            const double dz11 = h11 - h00;
            n = chrono::ChVector3d(dy * (dz01 - dz11), -dx * dz01, dx * dy);
        }
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
    std::array<double, 4> m_size = {1.0, 1.0, 1.0, 0.1};
    int m_nrow = 0;
    int m_ncol = 0;
    std::vector<float> m_data;
    chrono::ChVector3d m_pos = chrono::ChVector3d(0, 0, 0);
    float m_mu = 0.8f;
};

std::unique_ptr<chrono::vehicle::ChTerrain> ParseTerrain(const rj::Value& req) {
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
            return std::make_unique<AnalyticTerrain>(
                AnalyticTerrain::MakePlane(chrono::ChVector3d(0, 0, h), chrono::ChVector3d(0, 0, 1), mu));
        }
        const auto p = GetVec3(t, "point");
        const auto n = GetVec3(t, "normal");
        return std::make_unique<AnalyticTerrain>(AnalyticTerrain::MakePlane(p, n, mu));
    }

    if (type == "sinusoid") {
        const double base = GetNumber(t, "base");
        const double amp = GetNumber(t, "amp");
        const double freq = GetNumber(t, "freq");
        return std::make_unique<AnalyticTerrain>(AnalyticTerrain::MakeSinusoid(base, amp, freq, mu));
    }

    if (type == "hfield") {
        HFieldTerrain::Params p;
        p.mu = mu;
        p.size = GetVec4(t, "size");
        p.nrow = int(GetNumber(t, "nrow"));
        p.ncol = int(GetNumber(t, "ncol"));
        if (p.nrow <= 0 || p.ncol <= 0) {
            Throw("hfield: nrow/ncol must be > 0");
        }
        if (!t.HasMember("data") || !t["data"].IsArray()) {
            Throw("hfield: expected array 'data'");
        }
        const auto& a = t["data"];
        if (int(a.Size()) != p.nrow * p.ncol) {
            Throw("hfield: data length must match nrow*ncol");
        }
        p.data.resize(p.nrow * p.ncol);
        for (int i = 0; i < p.nrow * p.ncol; ++i) {
            if (!a[i].IsNumber()) {
                Throw("hfield: expected numeric data entries");
            }
            p.data[i] = float(a[i].GetDouble());
        }

        if (t.HasMember("pos")) {
            p.pos = GetVec3(t, "pos");
        }
        return std::make_unique<HFieldTerrain>(HFieldTerrain::Make(p));
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

chrono::vehicle::ChTireTestRig::Mode ParseTireTestRigMode(const std::string& s) {
    if (s == "suspend") {
        return chrono::vehicle::ChTireTestRig::Mode::SUSPEND;
    }
    if (s == "drop") {
        return chrono::vehicle::ChTireTestRig::Mode::DROP;
    }
    if (s == "test") {
        return chrono::vehicle::ChTireTestRig::Mode::TEST;
    }
    Throw(std::string("unsupported rig_mode: ") + s);
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
                method, *terrain, disc_center, disc_normal, disc_radius, width, area_dep, contact, depth, mu);

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
        } else if (cmd == "tire_test_rig") {
            // Chrono reference: chrono/src/chrono_vehicle/wheeled_vehicle/test_rig/ChTireTestRig.cpp
            // Chrono reference: chrono/src/demos/vehicle/test_rigs/demo_VEH_TireTestRig.cpp

            const std::string wheel_json = GetString(req, "wheel_json");
            const std::string tire_json = GetString(req, "tire_json");
            const std::string mode_str = GetString(req, "mode");
            const auto rig_mode = ParseTireTestRigMode(mode_str);

            const double dt = req.HasMember("dt") ? GetNumber(req, "dt") : 1e-3;
            const double t_end = req.HasMember("t_end") ? GetNumber(req, "t_end") : 2.0;
            const int decimate = req.HasMember("decimate") ? int(GetNumber(req, "decimate")) : 1;
            const double grav = req.HasMember("grav") ? GetNumber(req, "grav") : 9.8;
            const double normal_load = req.HasMember("normal_load") ? GetNumber(req, "normal_load") : 3000.0;
            const double camber = req.HasMember("camber") ? GetNumber(req, "camber") : 0.0;
            const double time_delay = req.HasMember("time_delay") ? GetNumber(req, "time_delay") : 1.0;

            const std::string coll_str =
                req.HasMember("collision_type") ? GetString(req, "collision_type") : "four_points";
            const auto coll_type = ParseCollisionType(coll_str);

            const double long_speed = req.HasMember("long_speed") ? GetNumber(req, "long_speed") : 0.2;
            const double ang_speed = req.HasMember("ang_speed") ? GetNumber(req, "ang_speed") : (10.0 * (2.0 * chrono::CH_PI / 60.0));
            const double sa_ampl =
                req.HasMember("slip_angle_ampl") ? GetNumber(req, "slip_angle_ampl") : (5.0 * (chrono::CH_PI / 180.0));
            const double sa_freq = req.HasMember("slip_angle_freq") ? GetNumber(req, "slip_angle_freq") : 0.2;
            const double sa_phase = req.HasMember("slip_angle_phase") ? GetNumber(req, "slip_angle_phase") : 0.0;
            const double sa_shift = req.HasMember("slip_angle_shift") ? GetNumber(req, "slip_angle_shift") : 0.0;

            // Terrain defaults match the Chrono demo.
            const double terrain_length = req.HasMember("terrain_length") ? GetNumber(req, "terrain_length") : 10.0;
            const double terrain_width = req.HasMember("terrain_width") ? GetNumber(req, "terrain_width") : 1.0;
            const float terrain_mu = static_cast<float>(req.HasMember("terrain_mu") ? GetNumber(req, "terrain_mu") : 0.8);
            // NOTE (Lukas): For Force-Element Tires with explicit/soft normal force laws like fiala, normal force is not determined by resititution and young modulus
            const float terrain_restitution =
                static_cast<float>(req.HasMember("terrain_restitution") ? GetNumber(req, "terrain_restitution") : 0.0);
            const float terrain_young =
                static_cast<float>(req.HasMember("terrain_young_modulus") ? GetNumber(req, "terrain_young_modulus") : 2e7);

            if (dt <= 0.0) {
                Throw("tire_test_rig: dt must be > 0");
            }
            if (t_end < 0.0) {
                Throw("tire_test_rig: t_end must be >= 0");
            }
            if (decimate <= 0) {
                Throw("tire_test_rig: decimate must be > 0");
            }

            auto wheel = chrono::vehicle::ReadWheelJSON(wheel_json);
            auto tire = chrono::vehicle::ReadTireJSON(tire_json);
            if (!wheel) {
                Throw(std::string("failed to read wheel JSON: ") + wheel_json);
            }
            if (!tire) {
                Throw(std::string("failed to read tire JSON: ") + tire_json);
            }

            chrono::ChSystemNSC sys;
            ConfigureChronoTireTestRigSystem(sys);
            chrono::vehicle::ChTireTestRig rig(wheel, tire, &sys);

            rig.SetGravitationalAcceleration(grav);
            rig.SetNormalLoad(normal_load);
            rig.SetCamberAngle(camber);
            rig.SetTireStepsize(dt);
            rig.SetTireCollisionType(coll_type);

            chrono::vehicle::ChTireTestRig::TerrainPatchSize size;
            size.length = terrain_length;
            size.width = terrain_width;
            chrono::vehicle::ChTireTestRig::TerrainParamsRigid params;
            params.friction = terrain_mu;
            params.restitution = terrain_restitution;
            params.Young_modulus = terrain_young;
            rig.SetTerrainRigid(size, params);

            if (rig_mode == chrono::vehicle::ChTireTestRig::Mode::TEST) {
                rig.SetLongSpeedFunction(chrono_types::make_shared<chrono::ChFunctionConst>(long_speed));
                rig.SetAngSpeedFunction(chrono_types::make_shared<chrono::ChFunctionConst>(ang_speed));
                rig.SetSlipAngleFunction(chrono_types::make_shared<chrono::ChFunctionSine>(sa_ampl, sa_freq, sa_phase, sa_shift));
                rig.SetTimeDelay(time_delay);
            } else {
                rig.SetTimeDelay(time_delay);
            }

            rig.Initialize(rig_mode);

            const int nsteps = int(std::round(t_end / dt));
            rj::Value samples(rj::kArrayType);
            samples.Reserve(std::max(0, (nsteps + decimate - 1) / decimate), alloc);

            for (int i = 0; i < nsteps; ++i) {
                rig.Advance(dt);

                if ((i % decimate) != 0) {
                    continue;
                }

                const double t = sys.GetChTime();
                const double slip = rig.GetLongitudinalSlip();
                const double slip_angle = rig.GetSlipAngle();
                const double camber_angle = rig.GetCamberAngle();

                const auto tf = rig.ReportTireForce();
                const auto spindle = rig.GetSpindle();

                rj::Value s(rj::kObjectType);
                s.AddMember("t", t, alloc);
                s.AddMember("slip", slip, alloc);
                s.AddMember("slip_angle", slip_angle, alloc);
                s.AddMember("camber_angle", camber_angle, alloc);

                rj::Value force_obj(rj::kObjectType);
                AddVec3(force_obj, alloc, "force", tf.force);
                AddVec3(force_obj, alloc, "moment", tf.moment);
                AddVec3(force_obj, alloc, "point", tf.point);
                s.AddMember("tire_force", force_obj, alloc);

                rj::Value spindle_obj(rj::kObjectType);
                AddVec3(spindle_obj, alloc, "pos", spindle->GetPos());
                AddVec3(spindle_obj, alloc, "vel", spindle->GetPosDt());
                AddVec3(spindle_obj, alloc, "omega_local", spindle->GetAngVelLocal());
                s.AddMember("spindle", spindle_obj, alloc);

                samples.PushBack(s, alloc);
            }

            resp.AddMember("dt", dt, alloc);
            resp.AddMember("t_end", t_end, alloc);
            resp.AddMember("decimate", decimate, alloc);
            resp.AddMember("ok", true, alloc);
            resp.AddMember("samples", samples, alloc);
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
