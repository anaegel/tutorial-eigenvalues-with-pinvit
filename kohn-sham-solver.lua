-- Load utility scripts (e.g. from from ugcore/scripts)
ug_load_script("ug_util.lua")
ug_load_script("util/refinement_util.lua")
-- ug_load_script("../modsim2/teaching_problems.lua")

-- initialize ug with the world dimension and the algebra type
InitUG(3, AlgebraType("CPU", 1));

numRefs = 5 -- 7 

-- Load a domain without initial refinements.
-- Refine the domain (parallel redistribution is handled internally!)
local gridName = "cube10_3d.ugx" -- grid name
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





-- External potential (nucleus-electron)
local nucleiDesc = {
    { Z = 2.0, pos = {0.01, 0.04, 0.01}}
}

function myExtPotential(x, y, z, t)
   
    local sum = 2.0    -- Add Shift so eigenvalues are positive.
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

-- 
evNumber = 1+4+9--10

-- Kohn-Sham configuration.
local ks = KohnShamData()
ks:init(approxSpace, evNumber)
ks:init_density()
ks:init_vxc(1) -- Select XC potential
ks:init_velectron()

-- Hartree potential (electron-electron)
local velectron = ks:get_velectron()
velectron:set(0.0)

-- XC Potential
local density = ks:get_density()
density:set(0.0)



disc = "fe"

local myElemDisc = ConvectionDiffusion("u", "Inner", disc)
myElemDisc:set_diffusion(1.0)

local veff=ScaleAddLinkerNumber()
veff:add(1.0, LuaUserNumber("myExtPotential"))       -- vext
-- veff:add(1.0, GridFunctionNumberData(velectron,"u")) -- vhartree
--veff:add(1.0, ks:get_vxc())                          -- vxc 
myElemDisc:set_reaction_rate(veff)
domainDisc:add(myElemDisc)

-- b) Boundary
local myDirichletBND = DirichletBoundary() 
myDirichletBND:add(0.0, "u", "Boundary")
domainDisc:add(myDirichletBND)




-- Create smoother. 
jac = Jacobi(0.66)             -- Jacobi (w/ damping)
gs  = GaussSeidel()            -- Gauss-Seidel (forward only)
sgs = SymmetricGaussSeidel()   -- Symmetric Gauss-Seidel (forward + backward)

local solverutil={}

local solverDesc = 
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
local gmg = util.solver.CreatePreconditioner(solverDesc.precond)
gmg:set_discretization(domainDisc)

-- Multigrid solver 
local lsolver = util.solver.CreateSolver(solverDesc, solverutil)
local convCheck = solverutil.convCheckDescs[1].instance



-- Define storage for eigenfunctions.
u = {}
for i=1,evNumber do
    u[i] = ks:get_wave_function(i-1)
    u[i]:set_random(-0.1, 0.1)
    domainDisc:adjust_solution(u[i]) -- Setze Randwerte
end

-- Define density as an (auxiliary) resulting quantity. 
function CreateDensityLinker(u, evNumber)
  local density = ScaleAddLinkerNumber()
  for i=1,evNumber do
    local phi = ExplicitGridFunctionValue(u[i], "u")
    density:add(phi,phi)
  end
  return density
end

local rho = CreateDensityLinker(u, evNumber)


-- Define matrix on left and right hand side
function CreatePINVITSolver(domainDisc)
    local A = MatrixOperator()
    domainDisc:assemble_stiffness_matrix(A, u[1])

    local B = MatrixOperator()
    domainDisc:assemble_mass_matrix(B, u[1])

    -- Define PINVIT.
    local evIterations = 50  -- number of iterations of the eigenvalue solver
    local evPrec = 1e-3       -- precision of the eigenvalue solver
    local evKeep = 8

    local pinvit = EigenSolver()
    pinvit:set_linear_operator_A(A)
    pinvit:set_linear_operator_B(B)
    pinvit:set_max_iterations(evIterations)
    pinvit:set_precision(evPrec)
    pinvit:set_preconditioner(gmg)
    pinvit:set_pinvit(2) -- ?
    pinvit:set_additional_eigenvectors_to_keep(evKeep)

    -- Add vectors.
    for i=1,evNumber do
        pinvit:add_vector(u[i])
    end

    return pinvit
end




-- Solve eigenvalue problem.
function SolveEigenvalueProblem(pinvit, domainDisc, u, evNumber)
  for i=1,evNumber do 
    domainDisc:adjust_solution(u[i])  
  end
  pinvit:apply()

  -- Print results.
  for i=1,evNumber do
    WriteGridFunctionToVTK(u[i], "ev_"..i)
  end
end



local pinvit= CreatePINVITSolver(domainDisc)
SolveEigenvalueProblem(pinvit, domainDisc, u, evNumber)

ks:update_density()
ks:update_vxc()

-- Print summary.
function PrintResults(u, ks, rho, filename)
  myVTK = VTKOutput()
  myVTK:select_element(rho, "density")
  myVTK:select_element(ks:get_density2(), "density user data")
  myVTK:select_element(ks:get_vxc(), "vxc")
  myVTK:print(filename, u[1])
end

PrintResults(u,ks,rho, "summary")


function CreatePoissonDisc(rho, myDirichletBND)
  local poissonDisc = DomainDiscretization(approxSpace)
  local myElemDisc = ConvectionDiffusion("u", "Inner", disc)
  local fourPi = 4.0 *3.1415926535897932384626433832795028
  myElemDisc:set_diffusion(1.0)
  myElemDisc:set_source(4.0*rho)
  poissonDisc:add(myElemDisc)
  poissonDisc:add(myDirichletBND)
  return poissonDisc
end

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

-- Loop until SCF convergence.
for i = 0,10 do
    print("SCF-Iteration: " .. i)
    
    local poissonDisc = CreatePoissonDisc(rho, myDirichletBND)

    ChangeParallelStorageTypeToAdditive3dCPU1(temp)
    PoissonSolver(velectron, temp, lsolver, poissonDisc)

    WriteGridFunctionToVTK(velectron, "velectron")
    
    local A = MatrixOperator()
    domainDisc:assemble_stiffness_matrix(A, u[1])

    local B = MatrixOperator()
    domainDisc:assemble_mass_matrix(B, u[1])

    pinvit:set_linear_operator_A(A)
    pinvit:set_linear_operator_B(B)
    
    SolveEigenvalueProblem(pinvit, domainDisc, u, evNumber)

    ks:update_density()
    ks:update_vxc()
end


