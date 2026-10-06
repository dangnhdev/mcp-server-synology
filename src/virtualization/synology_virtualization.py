"""Synology Virtual Machine Manager operations through its public API."""

import threading
import time
from typing import Any, Dict, Optional

from utils.synology_api import SynologyAPIClient


class SynologyVirtualization:
    """List, inspect, control, and delete Virtual Machine Manager guests."""

    guest_api = "SYNO.Virtualization.API.Guest"
    guest_action_api = "SYNO.Virtualization.API.Guest.Action"

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
            guests.append(
                {"guest_id": guest_id, "status": status if isinstance(status, str) else ""}
            )
        return guests, None

    @staticmethod
    def _failure(code: str, message: str, **data: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {"success": False, "error": {"code": code, "message": message}}
        if data:
            result["data"] = data
        return result
