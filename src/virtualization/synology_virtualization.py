"""Synology Virtual Machine Manager operations through its public API."""

import json
import threading
import time
from typing import Any, Dict, Optional

from utils.synology_api import SynologyAPIClient


class SynologyVirtualization:
    """Create, inspect, update, control, and delete VMM guests."""

    guest_api = "SYNO.Virtualization.API.Guest"
    guest_action_api = "SYNO.Virtualization.API.Guest.Action"
    storage_api = "SYNO.Virtualization.API.Storage"
    network_api = "SYNO.Virtualization.API.Network"
    image_api = "SYNO.Virtualization.API.Guest.Image"
    task_api = "SYNO.Virtualization.API.Task.Info"

    _RUNNING_STATES = frozenset({"running", "started", "online", "active"})
    _STOPPED_STATES = frozenset({"stopped", "shutdown", "offline", "inactive", "finished"})
    _AMBIGUOUS_REQUEST_ERRORS = frozenset(
        {"network_error", "invalid_response", "unknown_error"}
    )
    _SAFE_GUEST_FIELDS = (
        "guest_id",
        "guest_name",
        "name",
        "status",
        "state",
        "vcpu_num",
        "vram_size",
    )
    _VERIFY_ATTEMPTS = 8
    _VERIFY_INTERVAL_SECONDS = 0.5
    _MAX_CREATION_POLLS = 120
    _MAX_VM_DISKS = 8
    _MAX_VM_NETWORKS = 8

    def __init__(
        self,
        base_url: str,
        session_id: str,
        verify_ssl: bool = False,
        syno_token: Optional[str] = None,
    ):
        self._api = SynologyAPIClient(
            base_url,
            session_id,
            verify_ssl,
            syno_token=syno_token,
        )
        self._control_locks: Dict[str, Any] = {}
        self._control_locks_guard = threading.Lock()
        self._pending_create_names = set()
        self._pending_create_names_guard = threading.Lock()

    def list_virtual_machines(self, *, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Return the DSM VMM guest inventory."""
        result = self._api.post(self.guest_api, "list", 1, timeout=timeout)
        if not result.get("success"):
            return result
        _, error = self._guest_rows(result)
        if error:
            return error
        data = result["data"]
        root = "guests" if "guests" in data else "vms"
        return {
            "success": True,
            "data": {
                root: [
                    {key: guest[key] for key in self._SAFE_GUEST_FIELDS if key in guest}
                    for guest in data[root]
                ]
            },
        }

    def get_virtual_machine(self, guest_id: str) -> Dict[str, Any]:
        """Return details for one VMM guest, checking DSM returned the same ID."""
        normalized_id = guest_id.strip() if isinstance(guest_id, str) else ""
        if not normalized_id:
            return self._failure("invalid_guest_id", "guest_id must be a non-empty string")

        result = self._api.post(
            self.guest_api,
            "get",
            1,
            {"guest_id": normalized_id, "additional": "true"},
        )
        if not result.get("success"):
            return result

        data = result.get("data")
        response_id = data.get("guest_id") if isinstance(data, dict) else None
        if response_id != normalized_id:
            return self._failure(
                "invalid_response",
                "DSM returned a missing or mismatched guest_id for the requested virtual machine",
            )
        return {
            "success": True,
            "data": {
                key: data[key] for key in self._SAFE_GUEST_FIELDS if key in data
            },
        }

    def list_virtual_machine_resources(self) -> Dict[str, Any]:
        """List public VMM storage, network, and disk-image choices for VM creation."""
        resources = (
            (
                "storages",
                self.storage_api,
                ("storages", "storage"),
                ("storage_id", "id", "uuid"),
                ("storage_name", "name"),
            ),
            (
                "networks",
                self.network_api,
                ("networks", "network"),
                ("network_id", "id", "uuid"),
                ("network_name", "name"),
            ),
            (
                "disk_images",
                self.image_api,
                ("images", "image"),
                ("image_id", "id", "uuid"),
                ("image_name", "name"),
            ),
        )
        result_data = {}
        for section, api_name, roots, id_fields, name_fields in resources:
            rows, error = self._list_resource_rows(
                api_name, roots, id_fields, name_fields
            )
            if error:
                return error
            values = []
            for row in rows:
                if section == "disk_images" and row.get("type", "").casefold() != "disk":
                    continue
                item = {"id": row["id"], "name": row["name"]}
                if row.get("status"):
                    item["status"] = row["status"]
                if section == "disk_images" and row.get("type"):
                    item["type"] = row["type"]
                values.append(item)
            result_data[section] = values
        return {"success": True, "data": result_data}

    def create_virtual_machine(
        self,
        guest_name: str,
        storage_id: str,
        cpu_count: int,
        memory_mib: int,
        disks: Any,
        confirm: bool,
        network_ids: Optional[list] = None,
        description: str = "",
        auto_start: bool = False,
    ) -> Dict[str, Any]:
        """Create a VM through the public Guest API and verify task and config readback."""
        name = guest_name.strip() if isinstance(guest_name, str) else ""
        storage = storage_id.strip() if isinstance(storage_id, str) else ""
        if (
            not name
            or len(name) > 64
            or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in name)
            or not storage
            or storage_id != storage
            or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in storage)
        ):
            return self._failure(
                "invalid_input", "guest_name and storage_id must be valid non-empty strings"
            )
        if (
            not isinstance(cpu_count, int)
            or isinstance(cpu_count, bool)
            or not 1 <= cpu_count <= 64
            or not isinstance(memory_mib, int)
            or isinstance(memory_mib, bool)
            or not 128 <= memory_mib <= 1_048_576
        ):
            return self._failure(
                "invalid_input", "cpu_count must be 1-64 and memory_mib must be 128-1048576"
            )
        if not isinstance(description, str) or len(description) > 1_024 or "\x00" in description:
            return self._failure(
                "invalid_input", "description must be text no longer than 1024 characters"
            )
        if not isinstance(auto_start, bool):
            return self._failure("invalid_input", "auto_start must be a boolean")
        if confirm is not True:
            return self._failure(
                "confirmation_required", "Set confirm=true to authorize creating this virtual machine"
            )

        if not isinstance(disks, list) or not 1 <= len(disks) <= self._MAX_VM_DISKS:
            return self._failure(
                "invalid_disks", f"disks must contain 1-{self._MAX_VM_DISKS} disk specifications"
            )
        normalized_disks = []
        for index, disk in enumerate(disks):
            if not isinstance(disk, dict) or set(disk) not in (
                {"size_gib"},
                {"image_id"},
            ):
                return self._failure(
                    "invalid_disks",
                    "Each disk must specify exactly one of size_gib or image_id",
                    disk_index=index,
                )
            if "size_gib" in disk:
                size_gib = disk["size_gib"]
                if (
                    not isinstance(size_gib, int)
                    or isinstance(size_gib, bool)
                    or not 1 <= size_gib <= 1_048_576
                ):
                    return self._failure(
                        "invalid_disks",
                        "size_gib must be an integer between 1 and 1048576",
                        disk_index=index,
                    )
                normalized_disks.append({"size_gib": size_gib})
            else:
                image_id = disk["image_id"]
                if (
                    not isinstance(image_id, str)
                    or not image_id.strip()
                    or image_id != image_id.strip()
                    or any(ord(char) < 32 for char in image_id)
                ):
                    return self._failure(
                        "invalid_disks", "image_id must be a valid non-empty string", disk_index=index
                    )
                normalized_disks.append({"image_id": image_id})

        selected_networks = [""] if network_ids is None else network_ids
        if (
            not isinstance(selected_networks, list)
            or not 1 <= len(selected_networks) <= self._MAX_VM_NETWORKS
            or any(not isinstance(item, str) for item in selected_networks)
        ):
            return self._failure(
                "invalid_networks",
                f"network_ids must contain 1-{self._MAX_VM_NETWORKS} network IDs; use an empty string for an unconnected adapter",
            )
        normalized_networks = [item.strip() for item in selected_networks]
        if any(
            item != normalized or any(ord(char) < 32 for char in item)
            for item, normalized in zip(selected_networks, normalized_networks)
        ):
            return self._failure("invalid_networks", "network_ids contains an invalid value")

        lock_key = f"create-name:{name.casefold()}"
        with self._control_lock(lock_key):
            inventory = self.list_virtual_machines()
            if not inventory.get("success"):
                return inventory
            guests, error = self._guest_rows(inventory)
            if error:
                return error
            if any(
                guest["guest_name"].casefold() == name.casefold()
                for guest in guests
                if guest["guest_name"]
            ):
                with self._pending_create_names_guard:
                    self._pending_create_names.discard(name.casefold())
                return self._failure(
                    "name_conflict", "A virtual machine with this name already exists", guest_name=name
                )
            with self._pending_create_names_guard:
                if name.casefold() in self._pending_create_names:
                    return self._failure(
                        "submission_unverified",
                        "A previous create request for this name has an uncertain result. "
                        "Inspect VMM before trying again.",
                        guest_name=name,
                        verified=False,
                    )

            storages, error = self._list_resource_rows(
                self.storage_api,
                ("storages", "storage"),
                ("storage_id", "id", "uuid"),
                ("storage_name", "name"),
            )
            if error:
                return error
            selected_storage = next((item for item in storages if item["id"] == storage), None)
            if selected_storage is None or selected_storage.get("status", "").casefold() in {
                "error",
                "failed",
                "unhealthy",
            }:
                return self._failure(
                    "resource_conflict", "storage_id is missing or unavailable", storage_id=storage
                )

            connected_networks = {item for item in normalized_networks if item}
            if connected_networks:
                networks, error = self._list_resource_rows(
                    self.network_api,
                    ("networks", "network"),
                    ("network_id", "id", "uuid"),
                    ("network_name", "name"),
                )
                if error:
                    return error
                if not connected_networks.issubset({item["id"] for item in networks}):
                    return self._failure(
                        "resource_conflict", "One or more network IDs are no longer available"
                    )

            image_ids = {disk["image_id"] for disk in normalized_disks if "image_id" in disk}
            if image_ids:
                images, error = self._list_resource_rows(
                    self.image_api,
                    ("images", "image"),
                    ("image_id", "id", "uuid"),
                    ("image_name", "name"),
                )
                if error:
                    return error
                available_disk_images = {
                    item["id"] for item in images if item.get("type", "").casefold() == "disk"
                }
                if not image_ids.issubset(available_disk_images):
                    return self._failure(
                        "resource_conflict", "One or more disk image IDs are missing or are not disk images"
                    )

            with self._pending_create_names_guard:
                self._pending_create_names.add(name.casefold())

            request_disks = [
                {"create_type": 0, "vdisk_size": disk["size_gib"] * 1_024}
                if "size_gib" in disk
                else {"create_type": 1, "image_id": disk["image_id"]}
                for disk in normalized_disks
            ]
            request_result = self._api.post(
                self.guest_api,
                "create",
                1,
                {
                    "auto_clean_task": "false",
                    "storage_id": storage,
                    "vnics": json.dumps(
                        [{"network_id": item} for item in normalized_networks],
                        separators=(",", ":"),
                    ),
                    "vdisks": json.dumps(request_disks, separators=(",", ":")),
                    "guest_name": name,
                },
            )
            request_error = request_result.get("error")
            request_error_code = (
                request_error.get("code") if isinstance(request_error, dict) else None
            )
            if not request_result.get("success"):
                if request_error_code not in self._AMBIGUOUS_REQUEST_ERRORS:
                    with self._pending_create_names_guard:
                        self._pending_create_names.discard(name.casefold())
                    return request_result
                return self._failure(
                    "submission_unverified",
                    "DSM may have accepted the create request, but its task ID was not received. "
                    "Inspect VMM before retrying.",
                    guest_name=name,
                    verified=False,
                )

            response_data = request_result.get("data")
            task_id = response_data.get("task_id") if isinstance(response_data, dict) else None
            if not isinstance(task_id, str) or not task_id.strip():
                return self._failure(
                    "submission_unverified",
                    "DSM accepted the create request without returning a usable task ID. "
                    "Inspect VMM before retrying.",
                    guest_name=name,
                    verified=False,
                )
            task_id = task_id.strip()

            created_id = None
            for attempt in range(self._MAX_CREATION_POLLS):
                task_result = self._api.post(
                    self.task_api,
                    "get",
                    1,
                    {"task_id": task_id},
                    timeout=3,
                )
                if not task_result.get("success"):
                    return self._failure(
                        "creation_unverified",
                        "The create task was submitted, but its status could not be read. "
                        "Inspect VMM before retrying.",
                        guest_name=name,
                        task_id=task_id,
                        verified=False,
                    )
                task_data = task_result.get("data")
                if not isinstance(task_data, dict) or not isinstance(task_data.get("finish"), bool):
                    return self._failure(
                        "creation_unverified",
                        "DSM returned an invalid create task status. Inspect VMM before retrying.",
                        guest_name=name,
                        task_id=task_id,
                        verified=False,
                    )
                if task_data["finish"]:
                    task_info = task_data.get("task_info")
                    created_id = task_info.get("guest_id") if isinstance(task_info, dict) else None
                    if not isinstance(created_id, str) or not created_id.strip():
                        return self._failure(
                            "creation_unverified",
                            "DSM finished the create task without returning a guest ID. Inspect VMM before retrying.",
                            guest_name=name,
                            task_id=task_id,
                            verified=False,
                        )
                    created_id = created_id.strip()
                    break
                if attempt < self._MAX_CREATION_POLLS - 1:
                    time.sleep(self._VERIFY_INTERVAL_SECONDS)
            if created_id is None:
                return self._failure(
                    "creation_pending",
                    "The create task is still running. It was not submitted again; inspect VMM before retrying.",
                    guest_name=name,
                    task_id=task_id,
                    verified=False,
                )

            guest, error = self._read_guest_payload(created_id)
            if error:
                return self._failure(
                    "creation_unverified",
                    "DSM returned a guest ID, but the new VM details could not be read. "
                    "Inspect VMM before retrying.",
                    guest_id=created_id,
                    guest_name=name,
                    task_id=task_id,
                    verified=False,
                )
            actual_name = guest.get("guest_name")
            if actual_name != name:
                return self._failure(
                    "creation_unverified",
                    "The task guest ID did not match the requested VM name. Inspect VMM.",
                    guest_id=created_id,
                    guest_name=name,
                    task_id=task_id,
                    verified=False,
                )

            settings = {
                "guest_id": created_id,
                "new_guest_name": name,
                "description": description,
                "vcpu_num": str(cpu_count),
                "vram_size": str(memory_mib),
                "autorun": "2" if auto_start else "0",
            }
            settings_result = self._api.post(self.guest_api, "set", 1, settings)
            expected_settings = {
                "guest_name": name,
                "description": description,
                "vcpu_num": cpu_count,
                "vram_size": memory_mib,
                "autorun": 2 if auto_start else 0,
            }
            settings_verified = False
            current_guest = guest
            for attempt in range(self._VERIFY_ATTEMPTS):
                current_guest, read_error = self._read_guest_payload(created_id, timeout=3)
                if not read_error and self._guest_fields_match(current_guest, expected_settings):
                    settings_verified = True
                    break
                if attempt < self._VERIFY_ATTEMPTS - 1:
                    time.sleep(self._VERIFY_INTERVAL_SECONDS)

            hardware_verified = self._created_hardware_matches(
                current_guest, storage, normalized_disks, normalized_networks
            ) if settings_verified else False
            if settings_verified and hardware_verified:
                with self._pending_create_names_guard:
                    self._pending_create_names.discard(name.casefold())
                return {
                    "success": True,
                    "data": {
                        "guest_id": created_id,
                        "guest_name": name,
                        "task_id": task_id,
                        "created": True,
                        "settings_verified": True,
                        "hardware_verified": True,
                        "verified": True,
                        "request_accepted": bool(settings_result.get("success")),
                    },
                }
            settings_error = settings_result.get("error")
            return self._failure(
                "partial_creation",
                "The VM was created, but DSM did not confirm every requested setting and hardware item. "
                "Do not retry creation; inspect this VM in VMM.",
                guest_id=created_id,
                guest_name=name,
                task_id=task_id,
                created=True,
                settings_verified=settings_verified,
                hardware_verified=hardware_verified,
                request_accepted=(True if settings_result.get("success") else "unknown"),
                settings_error=settings_error if isinstance(settings_error, dict) else None,
                verified=False,
            )

    def update_virtual_machine(
        self,
        guest_id: str,
        confirm: bool,
        guest_name: Optional[str] = None,
        description: Optional[str] = None,
        cpu_count: Optional[int] = None,
        memory_mib: Optional[int] = None,
        auto_start: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update basic guest settings once and verify every requested field."""
        normalized_id = guest_id.strip() if isinstance(guest_id, str) else ""
        if not normalized_id:
            return self._failure("invalid_guest_id", "guest_id must be a non-empty string")
        if confirm is not True:
            return self._failure(
                "confirmation_required", "Set confirm=true to authorize updating this virtual machine"
            )
        if all(value is None for value in (guest_name, description, cpu_count, memory_mib, auto_start)):
            return self._failure(
                "invalid_input", "Provide at least one setting to update"
            )
        if guest_name is not None and (
            not isinstance(guest_name, str)
            or not guest_name.strip()
            or len(guest_name.strip()) > 64
            or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in guest_name)
        ):
            return self._failure("invalid_input", "guest_name must be a valid name up to 64 characters")
        if description is not None and (
            not isinstance(description, str)
            or len(description) > 1_024
            or "\x00" in description
        ):
            return self._failure(
                "invalid_input", "description must be text no longer than 1024 characters"
            )
        if cpu_count is not None and (
            not isinstance(cpu_count, int)
            or isinstance(cpu_count, bool)
            or not 1 <= cpu_count <= 64
        ):
            return self._failure("invalid_input", "cpu_count must be an integer from 1 to 64")
        if memory_mib is not None and (
            not isinstance(memory_mib, int)
            or isinstance(memory_mib, bool)
            or not 128 <= memory_mib <= 1_048_576
        ):
            return self._failure(
                "invalid_input", "memory_mib must be an integer from 128 to 1048576"
            )
        if auto_start is not None and not isinstance(auto_start, bool):
            return self._failure("invalid_input", "auto_start must be a boolean")

        with self._control_lock(normalized_id):
            inventory = self.list_virtual_machines()
            if not inventory.get("success"):
                return inventory
            guests, error = self._guest_rows(inventory)
            if error:
                return error
            matching = [guest for guest in guests if guest["guest_id"] == normalized_id]
            if len(matching) != 1:
                return self._failure(
                    "not_found" if not matching else "invalid_inventory",
                    "Virtual machine was not found" if not matching else "DSM returned duplicate guest IDs",
                    guest_id=normalized_id,
                )

            current, error = self._read_guest_payload(normalized_id)
            if error:
                return error
            expected = {}
            parameters = {"guest_id": normalized_id}

            if guest_name is not None:
                desired_name = guest_name.strip()
                if desired_name != current.get("guest_name"):
                    if any(
                        guest["guest_id"] != normalized_id
                        and guest["guest_name"].casefold() == desired_name.casefold()
                        for guest in guests
                        if guest["guest_name"]
                    ):
                        return self._failure(
                            "name_conflict", "Another virtual machine already uses this name", guest_name=desired_name
                        )
                    parameters["new_guest_name"] = desired_name
                    expected["guest_name"] = desired_name

            if description is not None and description != current.get("description"):
                parameters["description"] = description
                expected["description"] = description

            if cpu_count is not None and cpu_count != current.get("vcpu_num"):
                if matching[0]["status"].strip().lower() not in self._STOPPED_STATES:
                    return self._failure(
                        "state_conflict",
                        "Stop the virtual machine before changing its CPU count",
                        guest_id=normalized_id,
                        current_state=matching[0]["status"],
                    )
                parameters["vcpu_num"] = str(cpu_count)
                expected["vcpu_num"] = cpu_count

            if memory_mib is not None and memory_mib != current.get("vram_size"):
                if matching[0]["status"].strip().lower() not in self._STOPPED_STATES:
                    return self._failure(
                        "state_conflict",
                        "Stop the virtual machine before changing its memory",
                        guest_id=normalized_id,
                        current_state=matching[0]["status"],
                    )
                parameters["vram_size"] = str(memory_mib)
                expected["vram_size"] = memory_mib

            if auto_start is not None:
                current_autorun = current.get("autorun")
                desired_autorun = 2 if auto_start else 0
                if current_autorun != desired_autorun:
                    parameters["autorun"] = str(desired_autorun)
                    expected["autorun"] = desired_autorun

            if not expected:
                return {
                    "success": True,
                    "data": {
                        "guest_id": normalized_id,
                        "updated": [],
                        "verified": True,
                        "no_change": True,
                    },
                }

            request_result = self._api.post(self.guest_api, "set", 1, parameters)
            for attempt in range(self._VERIFY_ATTEMPTS):
                readback, error = self._read_guest_payload(normalized_id, timeout=3)
                if not error and self._guest_fields_match(readback, expected):
                    return {
                        "success": True,
                        "data": {
                            "guest_id": normalized_id,
                            "updated": sorted(expected),
                            "verified": True,
                            "request_accepted": bool(request_result.get("success")),
                        },
                    }
                if attempt < self._VERIFY_ATTEMPTS - 1:
                    time.sleep(self._VERIFY_INTERVAL_SECONDS)

            if not request_result.get("success"):
                return request_result
            return self._failure(
                "state_not_reached",
                "DSM did not report all requested settings. The update was not resubmitted; refresh the VM details.",
                guest_id=normalized_id,
                updated=sorted(expected),
                request_accepted=True,
                verified=False,
            )

    def control_virtual_machine(
        self,
        guest_id: str,
        action: str,
        confirm: bool,
    ) -> Dict[str, Any]:
        """Apply one power action after a state preflight and verify it with bounded readback.

        A response lost after submission is never retried. The inventory is
        reread to determine whether DSM completed the requested transition.
        """
        normalized_id = guest_id.strip() if isinstance(guest_id, str) else ""
        if not normalized_id:
            return self._failure("invalid_guest_id", "guest_id must be a non-empty string")
        if not isinstance(action, str) or action not in {"poweron", "shutdown", "poweroff"}:
            return self._failure(
                "invalid_action",
                "action must be one of: poweron, shutdown, poweroff",
            )
        if confirm is not True:
            return self._failure(
                "confirmation_required",
                "Set confirm=true to authorize this virtual machine power action",
            )

        lock = self._control_lock(normalized_id)
        with lock:
            inventory = self.list_virtual_machines()
            if not inventory.get("success"):
                return inventory
            guests, error = self._guest_rows(inventory)
            if error:
                return error

            matching = [guest for guest in guests if guest["guest_id"] == normalized_id]
            if len(matching) != 1:
                return self._failure(
                    "not_found" if not matching else "invalid_inventory",
                    "Virtual machine was not found" if not matching else "DSM returned duplicate guest IDs",
                    guest_id=normalized_id,
                )

            current_state = matching[0]["status"].strip().lower()
            expected_current = self._STOPPED_STATES if action == "poweron" else self._RUNNING_STATES
            if current_state not in expected_current:
                return self._failure(
                    "state_conflict",
                    f"Cannot run {action} while the virtual machine is in state '{current_state or 'unknown'}'",
                    guest_id=normalized_id,
                    current_state=current_state or "unknown",
                )

            request_result = self._api.post(
                self.guest_action_api,
                action,
                1,
                {"guest_id": normalized_id},
            )
            request_error = request_result.get("error")
            request_error_code = (
                request_error.get("code") if isinstance(request_error, dict) else None
            )
            if (
                not request_result.get("success")
                and request_error_code not in self._AMBIGUOUS_REQUEST_ERRORS
            ):
                return request_result

            expected_final = self._RUNNING_STATES if action == "poweron" else self._STOPPED_STATES
            final_state = "unknown"
            for attempt in range(self._VERIFY_ATTEMPTS):
                # Do not resubmit after an ambiguous transport failure. Read-only
                # inventory checks can confirm a completed transition.
                verification = self.list_virtual_machines(timeout=3)
                if not verification.get("success"):
                    if attempt == self._VERIFY_ATTEMPTS - 1:
                        return self._failure(
                            "verification_failed",
                            "The action may have been submitted, but DSM state could not be "
                            "read back. Inspect the virtual machine before retrying.",
                            guest_id=normalized_id,
                            action=action,
                            request_accepted=(
                                True if request_result.get("success") else "unknown"
                            ),
                            verified=False,
                        )
                else:
                    verified_guests, error = self._guest_rows(verification)
                    if error:
                        return self._failure(
                            "verification_failed",
                            "The action may have been submitted, but DSM returned an invalid "
                            "inventory. Inspect the virtual machine before retrying.",
                            guest_id=normalized_id,
                            action=action,
                            request_accepted=(
                                True if request_result.get("success") else "unknown"
                            ),
                            verified=False,
                        )
                    verified_guest = next(
                        (
                            guest
                            for guest in verified_guests
                            if guest["guest_id"] == normalized_id
                        ),
                        None,
                    )
                    final_state = (
                        verified_guest["status"].strip().lower() if verified_guest else "missing"
                    )
                    if final_state in expected_final:
                        return {
                            "success": True,
                            "data": {
                                "guest_id": normalized_id,
                                "action": action,
                                "status": final_state,
                                "request_accepted": (
                                    True if request_result.get("success") else "unknown"
                                ),
                                "verified": True,
                            },
                        }

                if attempt < self._VERIFY_ATTEMPTS - 1:
                    time.sleep(self._VERIFY_INTERVAL_SECONDS)

            return self._failure(
                "state_not_reached",
                "DSM did not report the requested final state. The action was not retried; "
                "refresh the inventory before taking another action.",
                guest_id=normalized_id,
                action=action,
                current_state=final_state or "unknown",
                request_accepted=(True if request_result.get("success") else "unknown"),
                verified=False,
            )

    def delete_virtual_machine(self, guest_id: str, confirm: bool) -> Dict[str, Any]:
        """Delete one stopped VMM guest and verify it is absent from inventory.

        The delete request is submitted at most once. If the response is
        ambiguous, read-only inventory checks determine whether it completed.
        """
        normalized_id = guest_id.strip() if isinstance(guest_id, str) else ""
        if not normalized_id:
            return self._failure("invalid_guest_id", "guest_id must be a non-empty string")
        if confirm is not True:
            return self._failure(
                "confirmation_required",
                "Set confirm=true to authorize deleting this virtual machine",
            )

        lock = self._control_lock(normalized_id)
        with lock:
            inventory = self.list_virtual_machines()
            if not inventory.get("success"):
                return inventory
            guests, error = self._guest_rows(inventory)
            if error:
                return error

            matching = [guest for guest in guests if guest["guest_id"] == normalized_id]
            if len(matching) != 1:
                return self._failure(
                    "not_found" if not matching else "invalid_inventory",
                    "Virtual machine was not found" if not matching else "DSM returned duplicate guest IDs",
                    guest_id=normalized_id,
                )

            current_state = matching[0]["status"].strip().lower()
            if current_state not in self._STOPPED_STATES:
                return self._failure(
                    "state_conflict",
                    "Only a stopped virtual machine can be deleted",
                    guest_id=normalized_id,
                    current_state=current_state or "unknown",
                )

            request_result = self._api.post(
                self.guest_api,
                "delete",
                1,
                {"guest_id": normalized_id},
            )
            request_error = request_result.get("error")
            request_error_code = (
                request_error.get("code") if isinstance(request_error, dict) else None
            )
            if (
                not request_result.get("success")
                and request_error_code not in self._AMBIGUOUS_REQUEST_ERRORS
            ):
                return request_result

            for attempt in range(self._VERIFY_ATTEMPTS):
                verification = self.list_virtual_machines(timeout=3)
                if not verification.get("success"):
                    if attempt == self._VERIFY_ATTEMPTS - 1:
                        return self._failure(
                            "verification_failed",
                            "The delete may have been submitted, but DSM state could not be "
                            "read back. Inspect the virtual machine before retrying.",
                            guest_id=normalized_id,
                            request_accepted=(
                                True if request_result.get("success") else "unknown"
                            ),
                            verified=False,
                        )
                else:
                    verified_guests, error = self._guest_rows(verification)
                    if error:
                        return self._failure(
                            "verification_failed",
                            "The delete may have been submitted, but DSM returned an invalid "
                            "inventory. Inspect the virtual machine before retrying.",
                            guest_id=normalized_id,
                            request_accepted=(
                                True if request_result.get("success") else "unknown"
                            ),
                            verified=False,
                        )
                    if not any(guest["guest_id"] == normalized_id for guest in verified_guests):
                        return {
                            "success": True,
                            "data": {
                                "guest_id": normalized_id,
                                "deleted": True,
                                "request_accepted": (
                                    True if request_result.get("success") else "unknown"
                                ),
                                "verified": True,
                            },
                        }

                if attempt < self._VERIFY_ATTEMPTS - 1:
                    time.sleep(self._VERIFY_INTERVAL_SECONDS)

            return self._failure(
                "state_not_reached",
                "DSM still reports the virtual machine. The delete request was not retried; "
                "refresh the inventory before taking another action.",
                guest_id=normalized_id,
                request_accepted=(True if request_result.get("success") else "unknown"),
                verified=False,
            )

    def _control_lock(self, guest_id: str) -> Any:
        """Serialize state checks and mutations for one guest."""
        with self._control_locks_guard:
            return self._control_locks.setdefault(guest_id, threading.Lock())

    def _list_resource_rows(
        self,
        api_name: str,
        roots: tuple[str, ...],
        id_fields: tuple[str, ...],
        name_fields: tuple[str, ...],
    ) -> tuple[list[dict], Optional[Dict[str, Any]]]:
        result = self._api.post(api_name, "list", 1)
        if not result.get("success"):
            return [], result
        data = result.get("data")
        accepted_roots = (*roots, "items")
        present_roots = [root for root in accepted_roots if isinstance(data, dict) and root in data]
        if len(present_roots) != 1 or not isinstance(data[present_roots[0]], list):
            return [], self._failure(
                "invalid_resource_inventory",
                f"DSM returned an invalid resource list for {api_name}",
            )

        resources = []
        seen_ids = set()
        for row in data[present_roots[0]]:
            if not isinstance(row, dict):
                return [], self._failure(
                    "invalid_resource_inventory",
                    f"DSM returned an invalid resource entry for {api_name}",
                )
            candidates = []
            for key in id_fields:
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
                elif isinstance(value, int) and not isinstance(value, bool):
                    candidates.append(str(value))
            candidates = sorted(set(candidates))
            if len(candidates) != 1 or candidates[0] in seen_ids:
                return [], self._failure(
                    "invalid_resource_inventory",
                    f"DSM returned a missing or duplicate resource ID for {api_name}",
                )
            resource_id = candidates[0]
            seen_ids.add(resource_id)
            name = next(
                (row[key].strip() for key in name_fields if isinstance(row.get(key), str) and row[key].strip()),
                resource_id,
            )
            status = next(
                (
                    row[key].strip()
                    for key in ("status", "state", "health")
                    if isinstance(row.get(key), str) and row[key].strip()
                ),
                "",
            )
            resource_type = next(
                (
                    row[key].strip()
                    for key in ("type", "image_type")
                    if isinstance(row.get(key), str) and row[key].strip()
                ),
                "",
            )
            resources.append(
                {"id": resource_id, "name": name, "status": status, "type": resource_type}
            )
        return resources, None

    def _read_guest_payload(
        self, guest_id: str, *, timeout: Optional[int] = None
    ) -> tuple[Optional[dict], Optional[Dict[str, Any]]]:
        result = self._api.post(
            self.guest_api,
            "get",
            1,
            {"guest_id": guest_id, "additional": "true"},
            timeout=timeout,
        )
        if not result.get("success"):
            return None, result
        data = result.get("data")
        if not isinstance(data, dict) or data.get("guest_id") != guest_id:
            return None, self._failure(
                "invalid_response",
                "DSM returned missing or mismatched guest details",
                guest_id=guest_id,
            )
        normalized = dict(data)
        for field in ("vcpu_num", "vram_size", "autorun"):
            value = normalized.get(field)
            if isinstance(value, str) and value.isdecimal():
                normalized[field] = int(value)
            elif isinstance(value, bool) or (value is not None and not isinstance(value, int)):
                return None, self._failure(
                    "invalid_response",
                    f"DSM returned an invalid {field} value for the virtual machine",
                    guest_id=guest_id,
                )
        return normalized, None

    @staticmethod
    def _guest_fields_match(guest: Optional[dict], expected: Dict[str, Any]) -> bool:
        if not isinstance(guest, dict):
            return False
        for field, expected_value in expected.items():
            current = guest.get(field)
            if field in {"vcpu_num", "vram_size", "autorun"}:
                if isinstance(current, str) and current.isdecimal():
                    current = int(current)
                if isinstance(current, bool) or not isinstance(current, int):
                    return False
            if current != expected_value:
                return False
        return True

    @staticmethod
    def _created_hardware_matches(
        guest: Optional[dict], storage_id: str, disks: list[dict], network_ids: list[str]
    ) -> bool:
        if (
            not isinstance(guest, dict)
            or guest.get("storage_id") != storage_id
            or any("image_id" in disk for disk in disks)
        ):
            # The public guest response does not identify the source image on a
            # cloned disk, so do not claim full hardware verification for it.
            return False

        def rows(value: Any) -> Optional[list]:
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return list(value.values())
            return None

        actual_disks = rows(guest.get("vdisks"))
        actual_networks = rows(guest.get("vnics"))
        if (
            actual_disks is None
            or actual_networks is None
            or len(actual_disks) != len(disks)
            or len(actual_networks) != len(network_ids)
            or any(not isinstance(item, dict) for item in (*actual_disks, *actual_networks))
        ):
            return False
        sizes = []
        disk_ids = set()
        for disk in actual_disks:
            disk_id = disk.get("vdisk_id")
            if not isinstance(disk_id, str) or not disk_id or disk_id in disk_ids:
                return False
            disk_ids.add(disk_id)
            size = disk.get("vdisk_size")
            if isinstance(size, str) and size.isdecimal():
                size = int(size)
            if isinstance(size, bool) or not isinstance(size, int):
                return False
            sizes.append(size)
        remaining_sizes = list(sizes)
        for requested in disks:
            requested_size = requested["size_gib"] * 1_024
            if requested_size not in remaining_sizes:
                return False
            remaining_sizes.remove(requested_size)
        network_interface_ids = [item.get("vnic_id") for item in actual_networks]
        if (
            any(not isinstance(item, str) or not item for item in network_interface_ids)
            or len(set(network_interface_ids)) != len(network_interface_ids)
        ):
            return False
        actual_network_ids = [item.get("network_id") for item in actual_networks]
        return all(isinstance(item, str) for item in actual_network_ids) and sorted(
            actual_network_ids
        ) == sorted(network_ids)

    @staticmethod
    def _guest_rows(result: Dict[str, Any]) -> tuple[list[dict], Optional[Dict[str, Any]]]:
        """Extract the public API inventory while rejecting ambiguous IDs."""
        data = result.get("data")
        rows = data.get("guests", data.get("vms")) if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return [], SynologyVirtualization._failure(
                "invalid_inventory", "DSM returned an invalid VMM guest inventory"
            )

        guests = []
        seen_ids = set()
        for row in rows:
            guest_id = row.get("guest_id") if isinstance(row, dict) else None
            if (
                not isinstance(guest_id, str)
                or not guest_id.strip()
                or guest_id != guest_id.strip()
                or guest_id in seen_ids
            ):
                return [], SynologyVirtualization._failure(
                    "invalid_inventory", "DSM returned a missing or duplicate guest_id"
                )
            seen_ids.add(guest_id)
            status = row.get("status")
            if not isinstance(status, str):
                status = row.get("state")
            guest_name = row.get("guest_name")
            if not isinstance(guest_name, str):
                guest_name = row.get("name")
            guests.append(
                {
                    "guest_id": guest_id,
                    "status": status if isinstance(status, str) else "",
                    "guest_name": guest_name if isinstance(guest_name, str) else "",
                }
            )
        return guests, None

    @staticmethod
    def _failure(code: str, message: str, **data: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {"success": False, "error": {"code": code, "message": message}}
        if data:
            result["data"] = data
        return result
