-- Load utility scripts (e.g. from from ugcore/scripts)
ug_load_script("ug_util.lua")
ug_load_script("util/refinement_util.lua")
-- ug_load_script("../modsim2/teaching_problems.lua")

-- initialize ug with the world dimension and the algebra type
InitUG(3, AlgebraType("CPU", 1));

numRefs = 4 -- 7 

-- Load a domain without initial refinements.
-- Refine the domain (parallel redistribution is handled internally!)
local gridName = "laplace_sample_grid_3d.ugx" -- grid name
local requiredSubsets = {"Inner", "Boundary"}

dom = util.CreateDomain(gridName, 0, requiredSubsets)
util.refinement.CreateRegularHierarchy(dom, numRefs, true)

-- Create approximation space.
approxSpace = ApproximationSpace(dom)
approxSpace:add_fct("u", "Lagrange", 1)
approxSpace:init_levels()
approxSpace:init_top_surface()
approxSpace:print_statistic()

-- Some auxiliary functions.
clock = CuckooClock()
OrderLex(approxSpace, "x")    -- order vertices (and dofs) lexicographically

-- Create discretization.
domainDisc = DomainDiscretization(approxSpace)
poissonDisc = DomainDiscretization(approxSpace)

-- Hartree potential (electron-electron)
velectron = GridFunction(approxSpace)
velectron:set(0.0)


-- External potential (nucleus-electron)
local nucleiDesc = {
    { Z = 1, pos = {0.3, 0.4, 0.1}}
}

function myExtPotential(x, y, z, t)
   
    local sum = 0.0
    for i,nucleus in ipairs(nucleiDesc) do
        local mycontrib = 0.0
        local rx = (x-nucleus.pos[1]);
        local ry = (y-nucleus.pos[2]); 
        local rz = (z-nucleus.pos[3]); 
        local dist2 = rx*rx+ry*ry+rz*rz;  
        sum = sum - nucleus.Z/math.sqrt(dist2);
    end
    print ("Ext="..sum)
    return sum
end

-- Debug output
temp = GridFunction(approxSpace)
temp:set(0.0)
Interpolate("myExtPotential", temp, "u")
WriteGridFunctionToVTK(temp, "vext")


-- XC Potential
density = GridFunction(approxSpace)
density:set(0.0)

xcPotential = XCFunctionalUserData(1)
xcPotential:set_density(GridFunctionNumberData(density, "u")) --

disc = "fe"

local myElemDisc = ConvectionDiffusion("u", "Inner", disc)
myElemDisc:set_diffusion(0.5)

local veff=ScaleAddLinkerNumber()
veff:add(1.0, LuaUserNumber("myExtPotential"))       -- vext
veff:add(1.0, GridFunctionNumberData(velectron,"u")) -- vhartree
veff:add(1.0, xcPotential)  -- vxc 
myElemDisc:set_reaction_rate(veff)
domainDisc:add(myElemDisc)

-- b) Boundary
myDirichletBND = DirichletBoundary() 
myDirichletBND:add(0.0, "u", "Boundary")
domainDisc:add(myDirichletBND)




-- Create smoother. 
jac = Jacobi(0.66)             -- Jacobi (w/ damping)
gs  = GaussSeidel()            -- Gauss-Seidel (forward only)
sgs = SymmetricGaussSeidel()   -- Symmetric Gauss-Seidel (forward + backward)

local solverutil={}

solverDesc = 
{
    type = "linear",  -- linear solver type ["bicgstab", "cg", "linear"]
    precond = {
        type        = "gmg",    -- preconditioner ["gmg", "ilu", "ilut", "jac", "gs", "sgs"]
        smoother    = "gs",     -- gmg-smoother ["ilu", "ilut", "jac", "gs", "sgs"]
        cycle       = "V",      -- gmg-cycle ["V", "F", "W"]
        preSmooth   = 1,        -- number presmoothing steps
        postSmooth  = 1,        -- number postsmoothing steps
        rap         = true,    -- computes RAP-product instead of assembling if true 
        
        baseLevel   = 0,        -- gmg - baselevel
        baseSolver  = "lu",
    
        approxSpace = approxSpace,
        
    },
    convCheck = "standard"
}

-- Multigrid preconditioner for PINVIT
gmg = util.solver.CreatePreconditioner(solverDesc.precond)
gmg:set_discretization(domainDisc)

-- Multigrid solver 
lsolver = util.solver.CreateSolver(solverDesc, solverutil)
convCheck = solverutil.convCheckDescs[1].instance

evNumber = 10

-- Define storage for eigenfunctions.
u = {}
for i=1,evNumber do
    u[i] = GridFunction(approxSpace)
    u[i]:set_random(-1.0, 1.0)
    domainDisc:adjust_solution(u[i]) -- Setze Randwerte
end

-- Define density as an (auxiliary) resulting quantity. 
rho = ScaleAddLinkerNumber()
for i=1,evNumber do
    phi = ExplicitGridFunctionValue(u[i], "u")
    rho:add(phi,phi)
end


-- Interpolate(rho, density, "u")

-- Define matrix on left and right hand side
A = MatrixOperator()
domainDisc:assemble_stiffness_matrix(A, u[1])

B = MatrixOperator()
domainDisc:assemble_mass_matrix(B, u[1])

-- Define PINVIT.
evIterations = 50  -- number of iterations of the eigenvalue solver
evPrec = 1e-3       -- precision of the eigenvalue solver
evKeep = 8

pinvit = EigenSolver()
pinvit:set_linear_operator_A(A)
pinvit:set_linear_operator_B(B)
pinvit:set_max_iterations(evIterations)
pinvit:set_precision(evPrec)
pinvit:set_preconditioner(gmg)
pinvit:set_pinvit(2) -- ?
pinvit:set_additional_eigenvectors_to_keep(evKeep)

-- Assign storage.
for i=1,evNumber do
    domainDisc:adjust_solution(u[i]) 
    pinvit:add_vector(u[i])
end

-- OPTIONAL: debug output
-- dbgWriter = GridFunctionDebugWriter(approxSpace)
-- pinvit:set_debug(dbgWriter)

-- Solve eigenvalue problem.
pinvit:apply()

-- Print results.
for i=1,evNumber do
    WriteGridFunctionToVTK(u[i], "ev_"..i)
end

-- Print density.
myVTK = VTKOutput()
myVTK:select_element(rho, "density")
myVTK:print("density", u[1])




function PoissonSolver(u, rhs, solver, poissonDisc)
   
    -- Assemble linear system
    local clock = CuckooClock()
    local A = AssembledLinearOperator(poissonDisc)
    poissonDisc:assemble_linear(A, rhs)
 
    -- Call solver
    clock:tic()
    solver:init(A, u) -- boundary values
    solver:apply_return_defect(u,rhs)
    
    print ("Poisson solver: "..clock:toc().." seconds.")
    WriteGridFunctionToVTK(rhs, "rho")
end    

local myElemDisc = ConvectionDiffusion("u", "Inner", disc)
local fourPi = 4.0 *3.1415926535897932384626433832795028
myElemDisc:set_diffusion(1.0)
myElemDisc:set_source(4.0*rho)
poissonDisc:add(myElemDisc)

poissonDisc:add(myDirichletBND)

ChangeParallelStorageTypeToAdditive3dCPU1(temp)
PoissonSolver(velectron, temp, lsolver, poissonDisc)

WriteGridFunctionToVTK(velectron, "velectron")




