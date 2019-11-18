package inventory

import (
	"github.com/markus-wa/demoinfocs-golang/common"
)
//TODO: Json marshal/unmarshaling for writing to file


type Equipment struct{
	Name string
	Count int
}

var equipmentList []Equipment
var equipmentKeys map[common.EquipmentElement]int // To be able to easier insert weapons correctly when retrieving weapons

//TODO: Make sure we have same array structure for image output as for this output,
// in regards to making the array indices the same for every equipment type
// Maps seem to have random order? Maybe need to hardcode this instead
func Init(){
	initEquipment()
	initEquipmentKeys()
}

func GetEquipmentList() []Equipment{
	returnList := make([]Equipment, len(equipmentList))
	copy(returnList, equipmentList)

	return returnList
}

func AddToList(list []Equipment, e common.EquipmentElement) []Equipment{
	list[equipmentKeys[e]].Count += 1
	return list
}

// Using a hardcoded array because we don't want the data to lie randomly everytime like it does with a map
func initEquipment(){
	newEquipment := Equipment{ Name: "AK-47", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "AUG", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "AWP", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "PP-Bizon", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "C4", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Desert Eagle", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Decoy Grenade", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Dual Berettas", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "FAMAS", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Five-SeveN", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Flashbang", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "G3SG1", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Galil AR", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Glock-18", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "HE Grenade", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "P2000", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Incendiary Grenade", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "M249", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "M4A4", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "MAC-10", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "MAG-7", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Molotov", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "MP7", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "MP5-SD", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "MP9", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Negev", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Nova", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "p250", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "P90", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Sawed-Off", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "SCAR-20", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "SG 553", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Smoke Grenade", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "SSG 08", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Zeus x27", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Tec-9", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "UMP-45", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "XM1014", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "M4A1", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "CZ75 Auto", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "USP-S", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "World", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "R8 Revolver", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Kevlar Vest", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Kevlar + Helmet", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Defuse Kit", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "Knife", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
	newEquipment = Equipment{ Name: "UNKNOWN", Count: 0,}
	equipmentList = append(equipmentList, newEquipment)
}

func initEquipmentKeys(){
	equipmentKeys = make(map[common.EquipmentElement]int)
	equipmentKeys[common.EqAK47] 			= 0
	equipmentKeys[common.EqAUG] 			= 1
	equipmentKeys[common.EqAWP] 			= 2
	equipmentKeys[common.EqBizon] 			= 3
	equipmentKeys[common.EqBomb] 			= 4
	equipmentKeys[common.EqDeagle] 			= 5
	equipmentKeys[common.EqDecoy] 			= 6
	equipmentKeys[common.EqDualBerettas] 	= 7
	equipmentKeys[common.EqFamas] 			= 8
	equipmentKeys[common.EqFiveSeven] 		= 9
	equipmentKeys[common.EqFlash] 			= 10
	equipmentKeys[common.EqG3SG1] 			= 11
	equipmentKeys[common.EqGalil] 			= 12
	equipmentKeys[common.EqGlock] 			= 13
	equipmentKeys[common.EqHE] 				= 14
	equipmentKeys[common.EqP2000] 			= 15
	equipmentKeys[common.EqIncendiary] 		= 16
	equipmentKeys[common.EqM249] 			= 17
	equipmentKeys[common.EqM4A4] 			= 18
	equipmentKeys[common.EqMac10] 			= 19
	equipmentKeys[common.EqSwag7] 			= 20
	equipmentKeys[common.EqMolotov] 		= 21
	equipmentKeys[common.EqMP7] 			= 22
	equipmentKeys[common.EqMP5] 			= 23
	equipmentKeys[common.EqMP9] 			= 24
	equipmentKeys[common.EqNegev] 			= 25
	equipmentKeys[common.EqNova] 			= 26
	equipmentKeys[common.EqP250] 			= 27
	equipmentKeys[common.EqP90] 			= 28
	equipmentKeys[common.EqSawedOff] 		= 29
	equipmentKeys[common.EqScar20] 			= 30
	equipmentKeys[common.EqSG553] 			= 31
	equipmentKeys[common.EqSmoke] 			= 32
	equipmentKeys[common.EqScout] 			= 33
	equipmentKeys[common.EqZeus] 			= 34
	equipmentKeys[common.EqTec9] 			= 35
	equipmentKeys[common.EqUMP] 			= 36
	equipmentKeys[common.EqXM1014] 			= 37
	equipmentKeys[common.EqM4A1] 			= 38
	equipmentKeys[common.EqCZ] 				= 39
	equipmentKeys[common.EqUSP] 			= 40
	equipmentKeys[common.EqWorld] 			= 41
	equipmentKeys[common.EqRevolver] 		= 42
	equipmentKeys[common.EqKevlar] 			= 43
	equipmentKeys[common.EqHelmet] 			= 44
	equipmentKeys[common.EqDefuseKit] 		= 45
	equipmentKeys[common.EqKnife] 			= 46
	equipmentKeys[common.EqUnknown] 		= 47
}

