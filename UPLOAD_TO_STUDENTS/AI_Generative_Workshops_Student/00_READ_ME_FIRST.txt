AI Generative Models Practical Workshops
========================================

START HERE
1. Keep this whole folder together.
2. Double-click Setup_Environment.bat once.
3. Double-click Check_Computer.bat. It creates hardware_profile.json.
4. Open the relevant workshop Word document in Docs.
5. Create each milestone Python file in the named workshop folder.
6. Run every milestone and confirm its checkpoint before moving on.

HARDWARE ADAPTATION
The same hardware_profile.json is read by all four workshops. Training workload
(batch size, subset size, model width, iterations/epochs and selected sampling
settings) is scaled automatically to hardware that PyTorch can actually use.
Do not manually edit hardware_profile.json during the class.

DATA
Workshops 1-3 use FashionMNIST. On the first normal run, torchvision downloads
it once into the datasets folder; later runs reuse the local copy. If classroom
internet is restricted, run the first data milestone once beforehand while online.
Workshop 4 generates its 2-D source and target data directly and needs no dataset download.
